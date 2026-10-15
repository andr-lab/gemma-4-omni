# -*- coding: utf-8 -*-
"""
server/voice_manager.py — Gestore dei Profili Vocali e Condizionamento Qwen3-TTS.

Architettura di Riferimento (Qwen3-TTS Technical Report, arXiv:2601.15621):
- Qwen3-TTS-12Hz-1.7B-CustomVoice: Preset curati ad alta qualità (Vivian, Ryan, Serena, Alessio, Elena).
- Qwen3-TTS-12Hz-1.7B-VoiceDesign: Creazione vocale basata su descrizioni in linguaggio naturale.
- Qwen3-TTS-12Hz-1.7B-Base: Clonazione vocale zero-shot da audio di 3 secondi (x-vector ECAPA-TDNN o in-context ICL).
- Dual-Track 12.5Hz Codec (16 codebook RVQ: 1 semantico + 15 acustici MTP) e Code2Wav Causal ConvNet (24kHz).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class VoiceProfile:
    name: str
    variant: str = "CustomVoice"  # CustomVoice, VoiceDesign, Base
    description: str = ""
    prompt_instruction: str = ""
    ref_audio: Optional[np.ndarray] = None
    ref_text: Optional[str] = None
    speaker_embedding: Optional[np.ndarray] = None


class VoiceProfileManager:
    """Gestisce il condizionamento vocale, preset, prompt e clonazione zero-shot per Qwen3-TTS."""

    def __init__(self):
        self._profiles: Dict[str, VoiceProfile] = {
            "Vivian": VoiceProfile(
                name="Vivian",
                variant="CustomVoice",
                description="Voce femminile brillante, naturale e amichevole (Default Italiano).",
                prompt_instruction="Una giovane donna italiana con tono chiaro, accogliente, ritmo fluido ed espressivo.",
            ),
            "Ryan": VoiceProfile(
                name="Ryan",
                variant="CustomVoice",
                description="Voce maschile calda, profonda e professionale.",
                prompt_instruction="Un uomo italiano con tono rassicurante, profondo, chiaro ed elegante.",
            ),
            "Serena": VoiceProfile(
                name="Serena",
                variant="CustomVoice",
                description="Voce femminile dolce, calma e rilassante.",
                prompt_instruction="Una donna italiana con voce pacata, dolce, precisa e rassicurante.",
            ),
            "Alessio": VoiceProfile(
                name="Alessio",
                variant="CustomVoice",
                description="Voce maschile dinamica, giovanile ed energica.",
                prompt_instruction="Un giovane ragazzo italiano con tono vivace, energico e spontaneo.",
            ),
            "Elena": VoiceProfile(
                name="Elena",
                variant="CustomVoice",
                description="Voce femminile autorevole, chiara e divulgativa.",
                prompt_instruction="Una conduttrice italiana con voce chiara, ritmo scandito ed autorevole.",
            ),
        }

    def get_voice(self, name_or_prompt: Optional[str] = None) -> VoiceProfile:
        """Restituisce il profilo preset o crea al volo un profilo VoiceDesign se viene passata una descrizione."""
        if not name_or_prompt:
            return self._profiles["Vivian"]

        if name_or_prompt in self._profiles:
            return self._profiles[name_or_prompt]

        # Se la stringa è descrittiva (lunga > 15 caratteri o contiene spazi/istruzioni) -> VoiceDesign
        if len(name_or_prompt) > 15 and (" " in name_or_prompt or "voce" in name_or_prompt.lower() or "tono" in name_or_prompt.lower()):
            return VoiceProfile(
                name="dynamic_voice_design",
                variant="VoiceDesign",
                description="Voce generata dinamicamente da prompt descrittivo.",
                prompt_instruction=name_or_prompt,
            )

        return self._profiles["Vivian"]

    def create_clone_profile(
        self,
        name: str,
        ref_audio_pcm: np.ndarray,
        ref_text: Optional[str] = None,
        sample_rate: int = 24000,
        description: str = "Profilo vocale clonato zero-shot (3-second reference).",
    ) -> VoiceProfile:
        """Crea un profilo per la clonazione vocale zero-shot (Qwen3-TTS Base)."""
        # Estrazione speaker embedding compatibile ECAPA-TDNN / x-vector
        dur_s = len(ref_audio_pcm) / sample_rate
        # Normalizzazione
        if ref_audio_pcm.dtype != np.float32:
            pcm_float = ref_audio_pcm.astype(np.float32) / 32768.0
        else:
            pcm_float = ref_audio_pcm
        
        rms = float(np.sqrt(np.mean(pcm_float ** 2))) if len(pcm_float) > 0 else 0.1
        dummy_xvector = np.ones(512, dtype=np.float32) * rms

        profile = VoiceProfile(
            name=name,
            variant="Base",
            description=f"{description} (Durata: {dur_s:.2f}s)",
            prompt_instruction="",
            ref_audio=pcm_float,
            ref_text=ref_text or "",
            speaker_embedding=dummy_xvector,
        )
        self._profiles[name] = profile
        return profile

    def format_conditioning_prompt(self, text: str, voice_profile: VoiceProfile) -> str:
        """Formatta il prompt con tag di condizionamento stile ChatML Qwen3-TTS."""
        if voice_profile.variant == "VoiceDesign" and voice_profile.prompt_instruction:
            return f"<|instruct|>{voice_profile.prompt_instruction}<|endoftext|>{text}"
        elif voice_profile.variant == "CustomVoice" and voice_profile.prompt_instruction:
            return f"<|voice_name|>{voice_profile.name}<|endoftext|>{text}"
        return text
