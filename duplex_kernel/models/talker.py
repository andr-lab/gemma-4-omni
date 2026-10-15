# -*- coding: utf-8 -*-
"""
duplex_kernel/models/talker.py — Qwen3-TTS Base Talker + Code2Wav.

The Base model is used in x-vector-only voice-clone mode: a fixed reference
waveform is converted once at model load into the official speaker embedding.
Only that small embedding survives model loading; the speaker encoder does not
remain part of the hot-path object.

Code2Wav streaming uses the decoder's real 72-frame sliding attention window.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.machinery
import logging
import math
import os
import sys
import types
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np
import torch

from duplex_kernel.qwen_conditioning import QWEN_AUDIO_CONDITIONING_RECIPE

PINNED_QWEN_TALKER_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"

logger = logging.getLogger(
    "duplex_kernel.talker"
)


@contextmanager
def capture_code2wav_preclamp(
    code2wav: torch.nn.Module,
    callback: Callable[[torch.Tensor], Any],
):
    """Observe or replace Code2Wav's final waveform immediately before clamp.

    qwen-tts 0.1.1's TokenizerV2 decoder has the concrete structure
    ``model.decoder.decoder[-1]``: the final ``decoder`` is a ModuleList whose
    last block produces ``wav`` immediately before ``wav.clamp(-1, 1)``.
    Hook only that block for the duration of one decode call. A callback may
    return a replacement tensor; returning ``None`` leaves the output intact.
    """
    decoder = getattr(code2wav, "decoder", None)
    blocks = getattr(decoder, "decoder", None)
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) == 0:
        raise RuntimeError(
            "Code2Wav pre-clamp hook contract mismatch: expected "
            "code2wav.decoder.decoder to be a nonempty ModuleList"
        )
    final_block = blocks[-1]

    def capture_output(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> torch.Tensor:
        if not isinstance(output, torch.Tensor):
            raise TypeError("Code2Wav final decoder block must return a Tensor")
        replacement = callback(output)
        if replacement is None:
            return output
        if not isinstance(replacement, torch.Tensor):
            raise TypeError("Code2Wav pre-clamp callback must return Tensor or None")
        if (
            replacement.shape != output.shape
            or replacement.device != output.device
            or replacement.dtype != output.dtype
        ):
            raise ValueError(
                "Code2Wav pre-clamp replacement must preserve shape, device, and dtype"
            )
        return replacement

    handle = final_block.register_forward_hook(capture_output)
    try:
        yield
    finally:
        handle.remove()


def rescale_code2wav_preclamp(
    waveform: torch.Tensor,
    *,
    max_peak: float = 0.95,
) -> torch.Tensor:
    """Apply opt-in scalar headroom before Qwen's hard output clamp."""
    if not isinstance(waveform, torch.Tensor) or not waveform.is_floating_point():
        raise TypeError("pre-clamp waveform must be a floating-point Tensor")
    if not math.isfinite(float(max_peak)) or not 0.0 < float(max_peak) < 1.0:
        raise ValueError("max_peak must be finite and between 0 and 1")
    if waveform.numel() == 0:
        return waveform
    peak = waveform.detach().float().abs().amax()
    if not bool(torch.isfinite(peak).item()):
        raise ValueError("pre-clamp waveform peak must be finite")
    scale = torch.clamp(peak / float(max_peak), min=1.0)
    return waveform / scale.to(dtype=waveform.dtype)


def _install_shims(
    workspace: str,
) -> None:
    """Shim torchaudio + ROPE for qwen_tts under Transformers 5.x."""
    try:
        import torchaudio  # noqa: F401
    except Exception:
        module = types.ModuleType(
            "torchaudio"
        )
        module.__spec__ = (
            importlib.machinery.ModuleSpec(
                "torchaudio",
                None,
            )
        )

        sys.modules[
            "torchaudio"
        ] = module

        sys.modules[
            "torchaudio._extension"
        ] = types.ModuleType(
            "torchaudio._extension"
        )

        sys.modules[
            "torchaudio.compliance"
        ] = types.ModuleType(
            "torchaudio.compliance"
        )

        sys.modules[
            "torchaudio.compliance.kaldi"
        ] = types.ModuleType(
            "torchaudio.compliance.kaldi"
        )

    sys.modules[
        "sox"
    ] = types.ModuleType(
        "sox"
    )

    import transformers.modeling_rope_utils as rope_utils

    def _default_rope_init(
        config,
        device=None,
        seq_len=None,
        **kwargs,
    ):
        del seq_len, kwargs

        base = getattr(
            config,
            "rope_theta",
            10000.0,
        )

        partial_rotary_factor = getattr(
            config,
            "partial_rotary_factor",
            1.0,
        )

        head_dim = getattr(
            config,
            "head_dim",
            (
                config.hidden_size
                // config.num_attention_heads
            ),
        )

        dim = int(
            head_dim
            * partial_rotary_factor
        )

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.int64,
                )
                .float()
                .to(device)
                / dim
            )
        )

        return inv_freq, 1.0

    rope_utils.ROPE_INIT_FUNCTIONS[
        "default"
    ] = _default_rope_init

    rope_utils.ROPE_INIT_FUNCTIONS[
        "interleaved"
    ] = _default_rope_init

    sys.path.insert(
        0,
        str(workspace),
    )

    sys.path.insert(
        0,
        str(
            os.path.join(
                workspace,
                "qwen3tts_src",
            )
        ),
    )



def _rebuild_code2wav_rotary_buffers(
    code2wav_model: torch.nn.Module,
) -> int:
    """Recompute Qwen decoder RoPE buffers after HF meta-device loading.

    Transformers 5.5 materializes non-persistent buffers with ``empty_like``
    when the model was initialized on ``meta``. Code2Wav's ``inv_freq`` is
    non-persistent, so it is not restored from the checkpoint. Rebuild it from
    the same config-bound initializer used by the Qwen module before serving.
    """
    rebuilt = 0
    rotary_class_name = "Qwen3TTSTokenizerV2DecoderRotatoryEmbedding"
    for module_name, module in code2wav_model.named_modules():
        if type(module).__name__ != rotary_class_name:
            continue
        if "inv_freq" not in module._buffers:
            raise RuntimeError(
                f"Code2Wav RoPE module {module_name} has no registered inv_freq buffer"
            )
        current = module._buffers["inv_freq"]
        if current.is_meta:
            raise RuntimeError(
                f"Code2Wav RoPE buffer {module_name}.inv_freq is still on meta"
            )
        rope_init_fn = getattr(module, "rope_init_fn", None)
        config = getattr(module, "config", None)
        if not callable(rope_init_fn) or config is None:
            raise RuntimeError(
                f"Code2Wav RoPE module {module_name} lacks its config initializer"
            )

        inv_freq, attention_scaling = rope_init_fn(config, device=current.device)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        rotary_dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        expected_numel = (rotary_dim + 1) // 2
        if not isinstance(inv_freq, torch.Tensor) or inv_freq.ndim != 1:
            raise RuntimeError(
                f"Code2Wav RoPE initializer returned invalid inv_freq for {module_name}"
            )
        if inv_freq.numel() != expected_numel or inv_freq.device != current.device:
            raise RuntimeError(
                f"Code2Wav RoPE initializer shape/device mismatch for {module_name}: "
                f"shape={tuple(inv_freq.shape)} expected_numel={expected_numel} "
                f"device={inv_freq.device} expected_device={current.device}"
            )
        if not bool(torch.isfinite(inv_freq).all().item()):
            raise RuntimeError(
                f"Code2Wav RoPE initializer returned non-finite inv_freq for {module_name}"
            )
        if isinstance(attention_scaling, torch.Tensor):
            scaling_finite = bool(torch.isfinite(attention_scaling).all().item())
        elif isinstance(attention_scaling, (int, float)):
            scaling_finite = math.isfinite(float(attention_scaling))
        else:
            scaling_finite = False
        if not scaling_finite:
            raise RuntimeError(
                f"Code2Wav RoPE initializer returned invalid attention scaling for {module_name}"
            )

        rope_type = getattr(module, "rope_type", None)
        if rope_type in {"default", "interleaved"}:
            if inv_freq.dtype != torch.float32:
                raise RuntimeError(
                    f"Code2Wav default RoPE inv_freq must be float32, got {inv_freq.dtype}"
                )
            if inv_freq.numel() and float(inv_freq.max().item()) > 1.0 + 1e-6:
                raise RuntimeError(
                    f"Code2Wav default RoPE inv_freq exceeds 1: {float(inv_freq.max().item())}"
                )

        module.register_buffer("inv_freq", inv_freq, persistent=False)
        module.original_inv_freq = inv_freq
        module.attention_scaling = attention_scaling
        rebuilt += 1

    if rebuilt == 0:
        raise RuntimeError(
            "Code2Wav decoder RoPE module was not found; refusing unverified buffers"
        )
    return rebuilt


def _sha256_file(
    path: str,
) -> str:
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _valid_sha256(
    value: str | None,
) -> bool:
    if not isinstance(
        value,
        str,
    ):
        return False

    text = value.strip().lower()

    return (
        len(text) == 64
        and all(
            char in "0123456789abcdef"
            for char in text
        )
    )


def _load_reference_audio_24k(
    path: str,
) -> np.ndarray:
    """Load a real mono speaker reference exactly once."""
    import soundfile as sf

    values, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    audio = np.asarray(
        values,
        dtype=np.float32,
    )

    if audio.ndim == 2:
        audio = audio.mean(
            axis=1,
            dtype=np.float32,
        )

    audio = audio.reshape(-1)

    if audio.size == 0:
        raise ValueError(
            "speaker reference WAV vuoto"
        )

    if not np.isfinite(
        audio
    ).all():
        raise ValueError(
            "speaker reference WAV "
            "contiene NaN/Inf"
        )

    target_rate = 24_000

    if int(sample_rate) != target_rate:
        from scipy.signal import (
            resample_poly,
        )

        divisor = math.gcd(
            int(sample_rate),
            target_rate,
        )

        audio = resample_poly(
            audio,
            target_rate // divisor,
            int(sample_rate) // divisor,
        ).astype(
            np.float32,
            copy=False,
        )

    if audio.size == 0:
        raise ValueError(
            "speaker reference vuoto "
            "dopo resample"
        )

    if not np.isfinite(
        audio
    ).all():
        raise ValueError(
            "speaker reference non finito "
            "dopo resample"
        )

    return np.ascontiguousarray(
        audio,
        dtype=np.float32,
    )


class TalkerModel:
    """Qwen3-TTS Base Talker plus TokenizerV2 decoder."""

    # Public provenance marker: this wrapper's generated-audio input contract
    # is versioned separately from the frozen Qwen/V9 weights.
    qwen_audio_conditioning_recipe = QWEN_AUDIO_CONDITIONING_RECIPE

    def __init__(
        self,
        talker: Any,
        proc: Any,
        code2wav: Any,
        cfg: Any,
        tcfg: Any,
        *,
        device: str,
        default_voice_conditioning:
            dict[str, Any] | None = None,
    ) -> None:
        self.talker = talker
        self.proc = proc
        self.code2wav = code2wav
        self.cfg = cfg
        self.tcfg = tcfg
        self.device = device

        self.default_voice_conditioning = (
            default_voice_conditioning
        )

    def resolve_voice_conditioning(
        self,
        voice_conditioning:
            dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if voice_conditioning is not None:
            return voice_conditioning

        return self.default_voice_conditioning

    def _speaker_embedding(
        self,
        voice_conditioning:
            dict[str, Any] | None,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        conditioning = (
            self.resolve_voice_conditioning(
                voice_conditioning
            )
        )

        if conditioning is None:
            return None

        if not isinstance(
            conditioning,
            dict,
        ):
            raise TypeError(
                "voice_conditioning "
                "deve essere dict"
            )

        mode = conditioning.get(
            "mode"
        )

        if mode != "x_vector_only":
            raise ValueError(
                "P7 supporta esclusivamente "
                "voice conditioning "
                "x_vector_only"
            )

        if conditioning.get(
            "ref_code"
        ) is not None:
            raise ValueError(
                "P7 x-vector-only non deve "
                "contenere ref_code"
            )

        value = conditioning.get(
            "speaker_embedding"
        )

        if value is None:
            raise ValueError(
                "speaker_embedding mancante"
            )

        if isinstance(
            value,
            torch.Tensor,
        ):
            tensor = value
        else:
            tensor = torch.as_tensor(
                value
            )

        if tensor.numel() != int(
            self.tcfg.hidden_size
        ):
            raise ValueError(
                "speaker_embedding dim "
                f"{tensor.numel()} != "
                f"{self.tcfg.hidden_size}"
            )

        tensor = (
            tensor
            .reshape(
                1,
                1,
                -1,
            )
            .to(
                device=self.device,
                dtype=dtype,
            )
        )

        if not bool(
            torch.isfinite(
                tensor
            ).all()
        ):
            raise ValueError(
                "speaker_embedding "
                "non finito"
            )

        return tensor

    def voice_conditioning_signature(
        self,
        voice_conditioning:
            dict[str, Any] | None,
    ) -> str | None:
        conditioning = (
            self.resolve_voice_conditioning(
                voice_conditioning
            )
        )

        if conditioning is None:
            return None

        embedding = conditioning.get(
            "speaker_embedding"
        )

        if embedding is None:
            raise ValueError(
                "speaker embedding mancante"
            )

        if isinstance(
            embedding,
            torch.Tensor,
        ):
            array = (
                embedding
                .detach()
                .float()
                .cpu()
                .contiguous()
                .numpy()
            )
        else:
            array = np.asarray(
                embedding,
                dtype=np.float32,
            )

        embedding_sha = (
            hashlib.sha256(
                np.ascontiguousarray(
                    array,
                    dtype=np.float32,
                ).tobytes()
            ).hexdigest()
        )

        reference_sha = str(
            conditioning.get(
                "reference_wav_sha256",
                "",
            )
        )

        return (
            "xvector:"
            f"{reference_sha}:"
            f"{embedding_sha}"
        )

    def build_prefill_parts(
        self,
        text: str,
        *,
        voice_conditioning:
            dict[str, Any] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build the Qwen stable/text/BOS prefill with official Base framing."""
        if not isinstance(
            text,
            str,
        ) or not text:
            raise ValueError(
                "Talker text vuoto"
            )

        talker = self.talker
        cfg = self.cfg
        tcfg = self.tcfg

        # Exact framing used by Qwen3TTSModel._build_assistant_text().
        assistant_text = (
            "<|im_start|>assistant\n"
            f"{text}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        input_ids = self.proc(
            text=assistant_text,
            return_tensors="pt",
        )["input_ids"].to(
            self.device
        )

        if (
            input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or input_ids.shape[1] <= 8
        ):
            raise RuntimeError(
                "Qwen assistant token layout "
                f"inatteso: {tuple(input_ids.shape)}"
            )

        # Upstream Qwen slicing contract:
        #   [:3]   -> "<|im_start|>assistant\n"
        #   [3:-5] -> synthesis text
        role_ids = input_ids[
            :,
            :3,
        ]

        raw_ids = input_ids[
            :,
            3:-5,
        ]

        if raw_ids.shape[1] <= 0:
            raise RuntimeError(
                "Qwen synthesis text "
                "senza token"
            )

        special_ids = torch.tensor(
            [[
                cfg.tts_bos_token_id,
                cfg.tts_eos_token_id,
                cfg.tts_pad_token_id,
            ]],
            device=self.device,
            dtype=input_ids.dtype,
        )

        (
            bos_e,
            eos_e,
            pad_e,
        ) = talker.text_projection(
            talker.get_text_embeddings()(
                special_ids
            )
        ).chunk(
            3,
            dim=1,
        )

        language_map = getattr(
            tcfg,
            "codec_language_id",
            {},
        )

        if "italian" not in language_map:
            raise RuntimeError(
                "Talker Base non espone "
                "codec_language_id['italian']"
            )

        language_id = int(
            language_map["italian"]
        )

        codec_prefill = torch.tensor(
            [[
                tcfg.codec_think_id,
                tcfg.codec_think_bos_id,
                language_id,
                tcfg.codec_think_eos_id,
            ]],
            device=self.device,
            dtype=input_ids.dtype,
        )

        codec_prefix_embedding = (
            talker.get_input_embeddings()(
                codec_prefill
            )
        )

        codec_tail_embedding = (
            talker.get_input_embeddings()(
                torch.tensor(
                    [[
                        tcfg.codec_pad_id,
                        tcfg.codec_bos_id,
                    ]],
                    device=self.device,
                    dtype=input_ids.dtype,
                )
            )
        )

        speaker_embedding = (
            self._speaker_embedding(
                voice_conditioning,
                dtype=codec_prefix_embedding.dtype,
            )
        )

        codec_parts = [
            codec_prefix_embedding,
        ]

        if speaker_embedding is not None:
            # Exact upstream Base-model position:
            #
            # codec think prefix
            # speaker embedding
            # codec pad / codec BOS
            codec_parts.append(
                speaker_embedding
            )

        codec_parts.append(
            codec_tail_embedding
        )

        codec_embedding = torch.cat(
            codec_parts,
            dim=1,
        )

        role_embedding = (
            talker.text_projection(
                talker.get_text_embeddings()(
                    role_ids
                )
            )
        )

        # Mirrors upstream:
        #
        # cat(
        #   tts_pad.expand(codec_len - 2),
        #   tts_bos,
        # ) + codec_embedding[:-1]
        #
        # The final codec BOS stays separate so the duplex kernel can insert
        # frame-addressable V9 conditioning immediately before it.
        icl_core = (
            torch.cat(
                (
                    pad_e.expand(
                        -1,
                        codec_embedding.shape[1]
                        - 2,
                        -1,
                    ),
                    bos_e,
                ),
                dim=1,
            )
            + codec_embedding[
                :,
                :-1,
            ]
        )

        icl_part = torch.cat(
            (
                role_embedding,
                icl_core,
            ),
            dim=1,
        )

        text_embedding = (
            talker.text_projection(
                talker.get_text_embeddings()(
                    raw_ids
                )
            )
        )

        codec_pad_embedding = (
            talker.get_input_embeddings()(
                torch.tensor(
                    [[
                        tcfg.codec_pad_id
                    ] * (
                        raw_ids.shape[1]
                        + 1
                    )],
                    device=self.device,
                    dtype=input_ids.dtype,
                )
            )
        )

        text_part = (
            torch.cat(
                (
                    text_embedding,
                    eos_e,
                ),
                dim=1,
            )
            + codec_pad_embedding
        )

        # Kept separate so FuturePrefixQwen can place:
        #
        # stable prefill | text | V9 frame slots | codec BOS
        bos_part = (
            pad_e
            + talker.get_input_embeddings()(
                torch.tensor(
                    [[
                        tcfg.codec_bos_id
                    ]],
                    device=self.device,
                    dtype=input_ids.dtype,
                )
            )
        )

        for name, value in (
            ("icl_part", icl_part),
            ("text_part", text_part),
            ("bos_part", bos_part),
            ("pad_e", pad_e),
        ):
            if not bool(
                torch.isfinite(
                    value
                ).all()
            ):
                raise RuntimeError(
                    f"{name} non finito"
                )

        return (
            icl_part,
            text_part,
            bos_part,
            pad_e,
        )

    def build_prefill_embeddings(
        self,
        text: str,
        *,
        voice_conditioning:
            dict[str, Any] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            icl_part,
            text_part,
            bos_part,
            pad_e,
        ) = self.build_prefill_parts(
            text,
            voice_conditioning=voice_conditioning,
        )

        return (
            torch.cat(
                (
                    icl_part,
                    text_part,
                ),
                dim=1,
            ),
            bos_part,
            pad_e,
        )

    def decode(
        self,
        codes: torch.Tensor,
        *,
        preclamp_callback: Callable[[torch.Tensor], Any] | None = None,
    ) -> torch.Tensor:
        """RVQ codes -> 24 kHz waveform."""
        if self.code2wav is None:
            raise RuntimeError("Code2Wav non caricato")
        if preclamp_callback is None:
            decoded = self.code2wav.decode(codes)
        else:
            with capture_code2wav_preclamp(self.code2wav, preclamp_callback):
                decoded = self.code2wav.decode(codes)
        return decoded.audio_values[0].detach().cpu()



def load_talker(
    workspace: str = "/workspace",
    *,
    device: str = "cuda",
    load_code2wav: bool = True,
    speaker_reference_wav:
        str | None = None,
    speaker_reference_sha256:
        str | None = None,
    qwen_talker_revision: str | None = None,
) -> TalkerModel:
    _install_shims(
        workspace
    )

    import qwen_tts.core.models as models
    import safetensors.torch as st

    from transformers import (
        AutoTokenizer,
    )

    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )

    snapshots = sorted(
        glob.glob(
            os.path.join(
                workspace,
                ".hf_cache/"
                "models--Qwen--"
                "Qwen3-TTS-12Hz-1.7B-Base/"
                "snapshots/*",
            )
        )
    )

    if not snapshots:
        raise FileNotFoundError(
            "Talker Qwen3-TTS Base "
            "non trovato"
        )

    expected_revision = (
        qwen_talker_revision
        or os.environ.get(
            "GEMMA4_OMNI_QWEN_TALKER_REVISION",
            PINNED_QWEN_TALKER_REVISION,
        )
    )
    snapshot = os.path.join(
        os.path.dirname(snapshots[0]),
        expected_revision,
    )
    if snapshot not in snapshots:
        raise FileNotFoundError(
            "snapshot Qwen3-TTS Base non attestata: "
            f"attesa={expected_revision}, disponibili={snapshots}"
        )

    config = (
        models.Qwen3TTSConfig
        .from_pretrained(
            snapshot
        )
    )

    talker_config = (
        config.talker_config
    )

    if not hasattr(
        talker_config,
        "pad_token_id",
    ):
        object.__setattr__(
            talker_config,
            "pad_token_id",
            None,
        )

    if (
        str(
            getattr(
                config,
                "tts_model_type",
                "",
            )
        ).lower()
        != "base"
    ):
        raise RuntimeError(
            "P7 richiede Qwen3-TTS "
            "Base model"
        )

    teacher = (
        models
        .Qwen3TTSForConditionalGeneration(
            config
        )
        .to(
            torch.bfloat16
        )
    )

    state_dict = st.load_file(
        os.path.join(
            snapshot,
            "model.safetensors",
        ),
        device="cpu",
    )

    teacher.load_state_dict(
        state_dict,
        strict=False,
    )

    del state_dict

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher = (
        teacher
        .to(
            device=device,
            dtype=torch.bfloat16,
        )
        .eval()
    )
    teacher.talker.eval().requires_grad_(False)
    teacher.talker.code_predictor.eval().requires_grad_(False)

    default_voice_conditioning = None

    if (
        speaker_reference_wav is None
    ) != (
        speaker_reference_sha256 is None
    ):
        raise ValueError(
            "speaker_reference_wav e "
            "speaker_reference_sha256 "
            "devono essere forniti insieme"
        )

    if speaker_reference_wav is not None:
        reference_path = os.path.abspath(
            os.path.expanduser(
                speaker_reference_wav
            )
        )

        if not os.path.isfile(
            reference_path
        ):
            raise FileNotFoundError(
                "speaker reference WAV "
                f"mancante: {reference_path}"
            )

        if not _valid_sha256(
            speaker_reference_sha256
        ):
            raise ValueError(
                "speaker reference SHA256 "
                "non valido"
            )

        actual_sha = _sha256_file(
            reference_path
        )

        if (
            actual_sha
            != speaker_reference_sha256
            .strip()
            .lower()
        ):
            raise RuntimeError(
                "speaker reference WAV "
                "SHA256 mismatch"
            )

        reference_audio = (
            _load_reference_audio_24k(
                reference_path
            )
        )

        encoder_rate = int(
            teacher
            .speaker_encoder_sample_rate
        )

        if encoder_rate != 24_000:
            raise RuntimeError(
                "Qwen Base speaker encoder "
                f"sample rate inatteso: "
                f"{encoder_rate}"
            )

        with torch.inference_mode():
            speaker_embedding = (
                teacher
                .extract_speaker_embedding(
                    audio=reference_audio,
                    sr=encoder_rate,
                )
            )

        if not isinstance(
            speaker_embedding,
            torch.Tensor,
        ):
            speaker_embedding = (
                torch.as_tensor(
                    speaker_embedding
                )
            )

        if (
            speaker_embedding.numel()
            != int(
                talker_config.hidden_size
            )
        ):
            raise RuntimeError(
                "speaker embedding dim "
                f"{speaker_embedding.numel()} "
                "!= Talker hidden size "
                f"{talker_config.hidden_size}"
            )

        if not bool(
            torch.isfinite(
                speaker_embedding
            ).all()
        ):
            raise RuntimeError(
                "speaker embedding "
                "non finito"
            )

        default_voice_conditioning = {
            "mode":
                "x_vector_only",

            "speaker_embedding":
                (
                    speaker_embedding
                    .detach()
                    .float()
                    .cpu()
                    .contiguous()
                ),

            "reference_wav_path":
                reference_path,

            "reference_wav_sha256":
                actual_sha,

            "ref_code":
                None,
        }

        logger.info(
            "[talker] Base speaker "
            "x-vector loaded; ref_sha=%s",
            actual_sha,
        )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            snapshot,
            use_fast=True,
        )
    )

    processor = (
        models.Qwen3TTSProcessor(
            tokenizer=tokenizer
        )
    )

    code2wav = None

    if load_code2wav:
        tokenizer_snapshots = sorted(
            glob.glob(
                os.path.join(
                    workspace,
                    ".hf_cache/"
                    "models--Qwen--"
                    "Qwen3-TTS-Tokenizer-12Hz/"
                    "snapshots/*",
                )
            )
        )

        if not tokenizer_snapshots:
            raise FileNotFoundError(
                "Code2Wav TokenizerV2 "
                "non trovato"
            )

        code2wav = (
            Qwen3TTSTokenizerV2Model
            .from_pretrained(
                tokenizer_snapshots[-1]
            )
            .to(device)
            .eval()
        )
        rebuilt_rope_count = _rebuild_code2wav_rotary_buffers(code2wav)
        logger.info(
            "[talker] Code2Wav RoPE buffers rebuilt from config count=%d",
            rebuilt_rope_count,
        )
        code2wav.eval().requires_grad_(False)

        try:
            window = int(
                code2wav
                .config
                .decoder_config
                .sliding_window
            )
        except (
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(
                "impossibile attestare "
                "Code2Wav sliding_window"
            ) from exc

        if window != 72:
            raise RuntimeError(
                "Code2Wav sliding_window "
                f"inattesa: {window} != 72"
            )

    logger.info(
        "[talker] Qwen3-TTS Base%s "
        "caricato",
        (
            " + Code2Wav"
            if code2wav is not None
            else " (Code2Wav escluso)"
        ),
    )

    # Keep only teacher.talker and the extracted x-vector.
    #
    # The heavy speaker encoder belongs to `teacher`; once this function
    # returns, only the referenced Talker submodule survives.
    loaded = TalkerModel(
        teacher.talker,
        processor,
        code2wav,
        teacher.config,
        talker_config,
        device=device,
        default_voice_conditioning=(
            default_voice_conditioning
        ),
    )
    loaded.qwen_talker_revision = os.path.basename(snapshot)
    loaded.qwen_snapshot_path = os.path.abspath(snapshot)
    imported_source = sys.modules.get("qwen_tts.core.models.modeling_qwen3_tts")
    loaded.qwen_source_path = (
        os.path.abspath(imported_source.__file__)
        if imported_source is not None and getattr(imported_source, "__file__", None)
        else None
    )
    return loaded


__all__ = [
    "PINNED_QWEN_TALKER_REVISION",
    "TalkerModel",
    "load_talker",
]
