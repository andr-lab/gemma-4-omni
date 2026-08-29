# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/input.py — Gemma4OmniPcmAppendBuffer e shim audio upstream.

Porting L2 (data plane audio-native): il client manda PCM (input_audio_buffer.
append) → il buffer accumula in chunk → al commit il payload audio viene
convertito in un WAV valido e iniettato nella history della sessione duplex →
il fallback chat lo inoltra al Thinker che fa STT upstream con l'encoder audio
nativo di Gemma4 (ADR-004/ADR-027: h_raw 12.5Hz dal USM-Conformer, NO Whisper).

Shim media connector: il chat fallback produce ``data:audio/pcm_f32le;base64``
(raw PCM float32) che il media connector di vLLM (pyav) non sa decodificare.
Qui aggiungiamo un header RIFF/WAV al volo in modo che pyav legga l'audio.
"""

from __future__ import annotations

import base64
import binascii
import io
import struct
from typing import Any

import numpy as np

SAMPLE_RATE_HZ = 16000
CHUNK_SAMPLES = 1600  # 100 ms @ 16 kHz (unità duplex ufficiale)


class Gemma4OmniPcmAppendBuffer:
    """Buffer PCM 16kHz f32le per le sessioni duplex L2.

    Accumula i chunk ``input_audio_buffer.append`` e, alla commit, produce un
    payload ``audio_url`` con WAV valido (RIFF header + PCM16) pronto per il
    media connector del fallback chat (STT upstream Thinker nativo).
    """

    def __init__(self) -> None:
        self._pending = bytearray()
        self._sample_rate_hz: int | None = None

    # -- PcmAppendBuffer contract -----------------------------------------
    @property
    def pending_byte_count(self) -> int:
        return len(self._pending)

    def clear(self) -> None:
        self._pending.clear()

    def clear_force_listen(self) -> None:
        # Nessun bit di force-listen server-side: no-op.
        return

    def has_pending(self) -> bool:
        return bool(self._pending)

    def has_reserved(self) -> bool:
        return False

    def prepare_append(
        self,
        payload: dict[str, Any],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool = True,
    ) -> Any | None:
        """Accumula un chunk audio (base64 f32le 16kHz)."""
        del operation_id, chunk_period_ms, allow_emit
        fmt = payload.get("format")
        audio = payload.get("audio")
        sr = payload.get("sample_rate_hz")
        if fmt != "pcm_f32le" or not isinstance(audio, str) or not isinstance(sr, int):
            # Payload non-audio (es. force_listen puro): nessun byte da bufferizzare.
            return None
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            return None
        if len(raw) % 4:
            return None
        if self._sample_rate_hz is None:
            self._sample_rate_hz = sr
        elif self._sample_rate_hz != sr:
            raise ValueError("Gemma4Omni duplex audio append sample_rate_hz changed within a session")
        self._pending.extend(raw)
        # Il fallback chat committa su input.commit: nessuna emissione parziale.
        return None

    def prepare_commit(self, *, operation_id: str, chunk_period_ms: int) -> Any:
        """Converte il PCM accumulato in un payload audio_url WAV valido."""
        del operation_id, chunk_period_ms
        payload = self.flush(chunk_period_ms=1000)
        if payload is None:
            return _NoopReservation(None)
        return _NoopReservation(payload)

    def flush(self, *, chunk_period_ms: int) -> dict[str, Any] | None:
        if not self._pending:
            return None
        raw = bytes(self._pending)
        self._pending.clear()
        sr = self._sample_rate_hz or SAMPLE_RATE_HZ
        wav_b64 = pcm_f32le_to_wav_b64(raw, sample_rate=sr)
        return {
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/wav;base64,{wav_b64}"},
            "sample_rate_hz": sr,
        }


class _NoopReservation:
    """Reservation minimal: il commit avviene subito (nessun rollback)."""

    __slots__ = ("payload", "byte_count")

    def __init__(self, payload: dict[str, Any] | None) -> None:
        self.payload = payload
        self.byte_count = 0

    def commit(self) -> None:
        return

    def rollback(self) -> None:
        return


# ---------------------------------------------------------------------------
# Conversione PCM f32le -> WAV PCM16 (RIFF) — valido per pyav (media connector)
# ---------------------------------------------------------------------------
def pcm_f32le_to_wav_b64(raw_f32le: bytes, *, sample_rate: int = 16000) -> str:
    """Converte PCM float32 little-endian in WAV PCM16 base64."""
    samples = np.frombuffer(raw_f32le, dtype="<f4")
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767.0).astype("<i2")
    data = pcm16.tobytes()
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits)
    header += b"data" + struct.pack("<I", len(data))
    return base64.b64encode(header + data).decode("ascii")


# ---------------------------------------------------------------------------
# Shim media connector: rende i data URI raw-PCM leggibili da pyav.
# Il fallback chat produce ``data:audio/pcm_f32le;base64,...`` (dal chunk
# convertito da convert_input_audio_with_rate) che pyav rifiuta. Questo shim
# intercetta quei payload e li riscrive come data:audio/wav.
# ---------------------------------------------------------------------------
_PCM_PATCHED = False


def install_audio_media_shim() -> None:
    """Patcha il media connector di vLLM per accettare raw PCM f32le.

    Il fallback chat produce ``data:audio/pcm_f32le;base64,...`` (dal chunk
    convertito da convert_input_audio_with_rate) che pyav rifiuta. Questo shim
    intercetta quei payload a livello di AudioMediaIO.load_base64 (che riceve
    media_type + bytes base64) e li riscrive come WAV PCM16 valido.
    """
    global _PCM_PATCHED
    if _PCM_PATCHED:
        return
    try:
        import vllm.multimodal.media.audio as _audio_mod
        import pybase64 as _b64

        _orig_load_base64 = _audio_mod.AudioMediaIO.load_base64
        _orig_load_bytes = _audio_mod.AudioMediaIO.load_bytes

        def _patched_load_base64(self, media_type: str, data: str):
            mt = (media_type or "").lower()
            if mt in {"audio/pcm_f32le", "pcm_f32le", "audio/pcm16", "pcm16",
                      "audio/pcm_s16le", "pcm_s16le", "audio/s16le", "s16le"}:
                try:
                    raw = _b64.b64decode(data, validate=True)
                    if mt in {"audio/pcm16", "pcm16", "audio/pcm_s16le", "pcm_s16le",
                              "audio/s16le", "s16le"} and len(raw) % 2 == 0:
                        pcm16 = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                        wav = pcm_f32le_to_wav_b64(pcm16.astype("<f4").tobytes())
                    elif len(raw) % 4 == 0:
                        wav = pcm_f32le_to_wav_b64(raw)
                    else:
                        return _orig_load_base64(self, media_type, data)
                    return _orig_load_bytes(self, _b64.b64decode(wav))
                except Exception:
                    pass
            return _orig_load_base64(self, media_type, data)

        _audio_mod.AudioMediaIO.load_base64 = _patched_load_base64
        _PCM_PATCHED = True
        print("[vllm_omni_plugin] audio media shim installata (raw PCM -> WAV)", flush=True)
    except Exception as exc:  # fail-soft
        print(f"[vllm_omni_plugin] audio media shim fallita: {exc}", flush=True)


__all__ = [
    "SAMPLE_RATE_HZ",
    "CHUNK_SAMPLES",
    "Gemma4OmniPcmAppendBuffer",
    "install_audio_media_shim",
    "pcm_f32le_to_wav_b64",
]
