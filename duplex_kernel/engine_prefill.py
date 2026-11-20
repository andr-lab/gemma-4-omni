from __future__ import annotations

import inspect
import os
import re
from numbers import Integral
from typing import Any

import torch
from duplex_kernel.config import (
    ARIA_BACKCHANNEL_FRAMES_PER_WORD,
    ARIA_BACKCHANNEL_MAX_FRAMES,
    ARIA_BACKCHANNEL_OVERHEAD_FRAMES,
    ARIA_FRAMES_PER_WORD,
    ARIA_MAX_FRAMES,
    ARIA_MIN_FRAMES,
    ARIA_OVERHEAD_FRAMES,
)

AUDIO_TEXT_PREFIX_CACHE_REUSE_ENV = "DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE"
AUDIO_THINKER_CACHE_BUCKETS = (512, 1024, 2048, 4096)


def _decode_generated_response_texts(
    tokenizer: Any,
    generated_token_ids: list[int] | tuple[int, ...],
) -> tuple[str, str]:
    """Return human-facing cleaned text and the exact protocol decode separately."""
    protocol_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    raw_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    display_text = re.sub(r"\s*<unused\d+>\s*", " ", raw_text)
    display_text = re.sub(r"<bos>|<eos>|<pad>|<end_of_turn>", " ", display_text)
    display_text = re.sub(r"[ \t]+", " ", display_text).strip()
    return display_text, protocol_text


def _cache_sequence_length(cache: Any) -> int:
    get_seq_length = getattr(cache, "get_seq_length", None)
    if not callable(get_seq_length):
        raise TypeError("Qwen Q1 requires a cache exposing get_seq_length()")
    length = get_seq_length()
    if isinstance(length, torch.Tensor):
        length = int(length.item())
    return int(length)


def select_audio_thinker_cache_bucket(
    prefill_tokens: int,
    output_budget: int,
    *,
    max_cache_len: int = AUDIO_THINKER_CACHE_BUCKETS[-1],
) -> int:
    """Return the smallest static-cache bucket covering one direct-audio turn.

    ``prefill_tokens`` must be the effective processor ``input_ids`` length,
    including audio placeholders. The bucket reserves the full requested
    generation budget; it never truncates either side to fit.
    """
    for name, value in (
        ("prefill_tokens", prefill_tokens),
        ("output_budget", output_budget),
        ("max_cache_len", max_cache_len),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} deve essere un intero")
    prefill_tokens = int(prefill_tokens)
    output_budget = int(output_budget)
    max_cache_len = int(max_cache_len)
    if prefill_tokens < 1:
        raise ValueError("prefill_tokens deve essere almeno 1")
    if output_budget < 1:
        raise ValueError("output_budget deve essere almeno 1")
    if max_cache_len < 1:
        raise ValueError("max_cache_len deve essere almeno 1")

    required_tokens = prefill_tokens + output_budget
    available = tuple(bucket for bucket in AUDIO_THINKER_CACHE_BUCKETS if bucket <= max_cache_len)
    largest_available = available[-1] if available else 0
    for bucket in available:
        if required_tokens <= bucket:
            return bucket
    raise ValueError(
        "prompt audio Thinker + budget supera i bucket StaticCache disponibili "
        f"({prefill_tokens} + {output_budget} = {required_tokens}; "
        f"massimo disponibile {largest_available}, massimo supportato "
        f"{AUDIO_THINKER_CACHE_BUCKETS[-1]})"
    )


def _estimate_talker_frame_budget(
    text: str,
    n_words: int,
    min_frames: int,
    *,
    short_streaming: bool,
    absolute_max_frames: int = ARIA_MAX_FRAMES,
) -> tuple[int, int, int]:
    """Return spoken units, digit-expansion units, and bounded CB0 frames.

    The caller supplies the ordinary word count to preserve established
    pacing. Each additional digit in a numeric token gets one spoken unit,
    since numeric IDs and codes are normally read digit by digit. Native EOS
    remains the stopping condition; this only supplies enough bounded budget.
    """
    if (
        isinstance(absolute_max_frames, bool)
        or not isinstance(absolute_max_frames, int)
        or absolute_max_frames <= 0
    ):
        raise ValueError("absolute_max_frames must be a positive integer")
    base_units = max(1, int(n_words))
    numeric_extra_units = 0
    for token in text.split():
        numeric_token = token.strip(".,;:!?()[]{}\"'“”‘’")
        if re.fullmatch(r"\d+(?:[-.,/]\d+)*", numeric_token):
            numeric_extra_units += max(0, sum(char.isdecimal() for char in numeric_token) - 1)
    spoken_units = base_units + numeric_extra_units
    if short_streaming:
        # Backchannels use an EOS-driven safety ceiling. A short text-derived
        # estimate can terminate valid speech before the model emits EOS.
        frame_budget = ARIA_BACKCHANNEL_MAX_FRAMES
    else:
        frame_budget = min(
            absolute_max_frames,
            max(
                int(min_frames),
                int(spoken_units * ARIA_FRAMES_PER_WORD) + ARIA_OVERHEAD_FRAMES,
            ),
        )
    return spoken_units, numeric_extra_units, frame_budget


def _native_q1_import_audio_frame_budget(
    text_frame_budget: int,
    *,
    live_qwen_text_stream: Any | None,
    live_frame_cap: int,
) -> int:
    """Reserve bounded headroom when live SAY text grows after Q1 import."""
    text_frame_budget = int(text_frame_budget)
    live_frame_cap = int(live_frame_cap)
    if text_frame_budget <= 0 or live_frame_cap <= 0:
        raise ValueError("native Q1 frame budgets devono essere positivi")
    if text_frame_budget > live_frame_cap:
        raise ValueError("text frame budget supera il cap live native Q1")
    return live_frame_cap if live_qwen_text_stream is not None else text_frame_budget


def _apply_live_frame_residual(
    adapter: Any,
    selected_response_hidden: torch.Tensor,
    qwen_audio_input: torch.Tensor,
    *,
    generation_step: int,
    tail_hold: bool,
) -> tuple[torch.Tensor, bool]:
    """Apply the already-selected live packet hidden at its global Q1 step."""
    if isinstance(generation_step, bool) or int(generation_step) < 0:
        raise ValueError("generation_step deve essere non negativo")
    if int(generation_step) == 0 and bool(getattr(adapter, "preserve_frame0", False)):
        return qwen_audio_input, bool(tail_hold)
    return adapter(selected_response_hidden, qwen_audio_input), bool(tail_hold)


def _audio_text_prefix_cache_reuse_from_env() -> bool:
    """Read the opt-in cross-response audio-prefix cache switch (default off)."""
    value = os.environ.get(AUDIO_TEXT_PREFIX_CACHE_REUSE_ENV)
    if value is None:
        return False
    value = value.strip()
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{AUDIO_TEXT_PREFIX_CACHE_REUSE_ENV} must be exactly '0' or '1'")


def _build_backchannel_token_trie(
    phrases: list[str],
    tokenizer: Any,
    *,
    eos_token_ids: set[int],
) -> dict[tuple[int, ...], frozenset[int]]:
    """Map each gold-phrase prefix to only its legal next Gemma token IDs.

    Phrase text comes from the train-split BACKCHANNEL targets bundled with
    the active checkpoint. Token IDs are always produced by the loaded
    tokenizer; there are no hand-authored IDs or fallback phrases.
    """
    eos = {int(token_id) for token_id in eos_token_ids}
    if not eos:
        raise ValueError("backchannel trie richiede almeno un token EOS reale")
    if not phrases:
        raise ValueError("backchannel trie vuoto: nessuna frase gold disponibile")

    prefixes: dict[tuple[int, ...], set[int]] = {}
    encoded_sequences: set[tuple[int, ...]] = set()
    for phrase in phrases:
        if not isinstance(phrase, str):
            raise TypeError("backchannel lexicon contiene una frase non testuale")
        phrase = phrase.strip()
        word_count = len(phrase.split())
        if not phrase or not 1 <= word_count <= 3 or "?" in phrase or "？" in phrase:
            raise ValueError(f"frase backchannel fuori contratto (1–3 parole, non domanda): {phrase!r}")
        encoded = tokenizer.encode(phrase, add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            if len(encoded) != 1:
                raise ValueError("tokenizer ha prodotto un batch inatteso per una frase")
            encoded = encoded[0]
        sequence = tuple(int(token_id) for token_id in encoded)
        if not sequence:
            raise ValueError(f"tokenizer non produce token per frase backchannel {phrase!r}")
        if any(token_id in eos or token_id < 0 for token_id in sequence):
            raise ValueError(f"frase backchannel contiene un token EOS/negativo: {phrase!r}")
        if sequence in encoded_sequences:
            continue
        encoded_sequences.add(sequence)
        for index, token_id in enumerate(sequence):
            prefix = sequence[:index]
            prefixes.setdefault(prefix, set()).add(token_id)
        prefixes.setdefault(sequence, set()).update(eos)

    if not encoded_sequences:
        raise ValueError("backchannel trie vuoto dopo la tokenizzazione")
    return {prefix: frozenset(next_ids) for prefix, next_ids in prefixes.items()}


def _resolve_talker_voice_conditioning(
    talker: Any,
    voice_conditioning: dict[str, Any] | None,
) -> dict[str, Any] | None:
    resolver = getattr(
        talker,
        "resolve_voice_conditioning",
        None,
    )
    if callable(resolver):
        return resolver(voice_conditioning)
    return voice_conditioning


def _build_talker_prefill_parts(
    talker: Any,
    text: str,
    voice_conditioning: dict[str, Any] | None,
    *,
    raw_text_token_ids: list[int] | tuple[int, ...] | None = None,
    include_text_eos: bool = True,
):
    builder = getattr(
        talker,
        "build_prefill_parts",
    )
    try:
        parameters = inspect.signature(
            builder
        ).parameters
    except (
        TypeError,
        ValueError,
    ):
        parameters = {}

    kwargs: dict[str, Any] = {}
    if "voice_conditioning" in parameters:
        kwargs["voice_conditioning"] = voice_conditioning
    elif voice_conditioning is not None:
        raise TypeError("Talker backend non supporta voice_conditioning")
    if "raw_text_token_ids" in parameters:
        kwargs["raw_text_token_ids"] = raw_text_token_ids
    elif raw_text_token_ids is not None:
        raise TypeError("Talker backend non supporta raw_text_token_ids live")
    if "include_text_eos" in parameters:
        kwargs["include_text_eos"] = include_text_eos
    elif not include_text_eos:
        raise TypeError("Talker backend non supporta prefill senza text EOS")
    if kwargs:
        return builder(text, **kwargs)
    if voice_conditioning is not None:
        raise TypeError("Talker backend non supporta voice_conditioning")
    return builder(text)


def _split_text_for_frame_segments(
    text: str,
    *,
    total_frames: int,
    segment_frames: int,
) -> tuple[str, ...]:
    """Partition a completed response across acoustic segment budgets.

    Segment rollovers reset Qwen's local audio history, so replaying the whole
    response text at each rollover would restart the utterance.  The pre-QLoRA
    runtime already has the complete Thinker text; assign contiguous word
    spans in proportion to each segment's acoustic budget instead.
    """
    if total_frames <= 0 or segment_frames <= 0:
        raise ValueError("frame budgets devono essere positivi")
    words = text.split()
    if not words:
        raise ValueError("text non può essere vuoto")
    segment_count = (int(total_frames) + int(segment_frames) - 1) // int(segment_frames)
    if segment_count == 1:
        return (text,)
    if segment_count > len(words):
        raise ValueError(
            "continuazione Qwen richiede almeno una parola per segmento: "
            f"segments={segment_count}, words={len(words)}"
        )

    result: list[str] = []
    word_start = 0
    for segment_index in range(segment_count):
        if segment_index == segment_count - 1:
            word_end = len(words)
        else:
            frame_end = min(
                int(total_frames),
                (segment_index + 1) * int(segment_frames),
            )
            rounded_end = (
                frame_end * len(words) + int(total_frames) // 2
            ) // int(total_frames)
            min_end = word_start + 1
            max_end = len(words) - (segment_count - segment_index - 1)
            word_end = min(max(rounded_end, min_end), max_end)
        result.append(" ".join(words[word_start:word_end]))
        word_start = word_end

    if word_start != len(words) or any(not part for part in result):
        raise RuntimeError("partizione testo/acoustic frame non contigua")
    return tuple(result)


def _validate_response_bundle_qlora_compatibility(
    *,
    bundle_enabled: bool,
    gemma_qlora_enabled: bool,
    native_control6_enabled: bool,
    response_lora_ablated: bool,
) -> None:
    """Only native Control6 with the bundle response LoRA ablated may stack adapters."""
    if bundle_enabled and gemma_qlora_enabled and not (
        native_control6_enabled and response_lora_ablated
    ):
        raise ValueError(
            "duplex response bundle + Gemma QLoRA is supported only with native Control6 "
            "and --ablate-response-lora"
        )
