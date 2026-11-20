import re

path = "duplex_kernel/engine.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Import ARIA_HARD_MAX_FRAMES
target_import = "from duplex_kernel.config import (\n    ARIA_FRAMES_PER_WORD,\n    ARIA_MAX_FRAMES,"
replacement_import = "from duplex_kernel.config import (\n    ARIA_FRAMES_PER_WORD,\n    ARIA_HARD_MAX_FRAMES,\n    ARIA_MAX_FRAMES,"

if target_import not in content:
    # try CRLF
    target_import = target_import.replace("\n", "\r\n")
    replacement_import = replacement_import.replace("\n", "\r\n")
assert target_import in content, "target_import not found"
content = content.replace(target_import, replacement_import, 1)

# 2. _estimate_talker_frame_budget
target_budget = """def _estimate_talker_frame_budget(
    text: str,
    n_words: int,
    min_frames: int,
    *,
    short_streaming: bool,
) -> tuple[int, int, int]:
    \"\"\"Return spoken units, digit-expansion units, and bounded CB0 frames.

    The caller supplies the ordinary word count to preserve established
    pacing. Each additional digit in a numeric token gets one spoken unit,
    since numeric IDs and codes are normally read digit by digit. Native EOS
    remains the stopping condition; this only supplies enough bounded budget.
    \"\"\"
    base_units = max(1, int(n_words))
    numeric_extra_units = 0
    for token in text.split():
        numeric_token = token.strip(".,;:!?()[]{}\\"'“”‘’")
        if re.fullmatch(r"\\d+(?:[-.,/]\\d+)*", numeric_token):
            numeric_extra_units += max(0, sum(char.isdecimal() for char in numeric_token) - 1)
    spoken_units = base_units + numeric_extra_units
    if short_streaming:
        # Backchannels use an EOS-driven safety ceiling. A short text-derived
        # estimate can terminate valid speech before the model emits EOS.
        frame_budget = ARIA_BACKCHANNEL_MAX_FRAMES
    else:
        frame_budget = min(
            ARIA_MAX_FRAMES,
            max(
                int(min_frames),
                int(spoken_units * ARIA_FRAMES_PER_WORD) + ARIA_OVERHEAD_FRAMES,
            ),
        )
    return spoken_units, numeric_extra_units, frame_budget"""

replacement_budget = """def _estimate_talker_frame_budget(
    text: str,
    n_words: int,
    min_frames: int,
    *,
    short_streaming: bool,
    max_frames: int | None = None,
) -> tuple[int, int, int]:
    \"\"\"Return spoken units, digit-expansion units, and bounded CB0 frames.

    The caller supplies the ordinary word count to preserve established
    pacing. Each additional digit in a numeric token gets one spoken unit,
    since numeric IDs and codes are normally read digit by digit. Native EOS
    remains the stopping condition; this only supplies enough bounded budget.
    \"\"\"
    base_units = max(1, int(n_words))
    numeric_extra_units = 0
    for token in text.split():
        numeric_token = token.strip(".,;:!?()[]{}\\"'“”‘’")
        if re.fullmatch(r"\\d+(?:[-.,/]\\d+)*", numeric_token):
            numeric_extra_units += max(0, sum(char.isdecimal() for char in numeric_token) - 1)
    spoken_units = base_units + numeric_extra_units
    if short_streaming:
        # Backchannels use an EOS-driven safety ceiling. A short text-derived
        # estimate can terminate valid speech before the model emits EOS.
        frame_budget = ARIA_BACKCHANNEL_MAX_FRAMES
    else:
        hard_cap = (
            ARIA_MAX_FRAMES
            if max_frames is None
            else min(
                ARIA_HARD_MAX_FRAMES,
                max(ARIA_MAX_FRAMES, int(max_frames)),
            )
        )
        frame_budget = min(
            hard_cap,
            max(
                int(min_frames),
                int(spoken_units * ARIA_FRAMES_PER_WORD) + ARIA_OVERHEAD_FRAMES,
            ),
        )
    return spoken_units, numeric_extra_units, frame_budget"""

if target_budget not in content:
    target_budget = target_budget.replace("\n", "\r\n")
    replacement_budget = replacement_budget.replace("\n", "\r\n")
assert target_budget in content, "target_budget not found"
content = content.replace(target_budget, replacement_budget, 1)

# 3. PipelineEngine.talk configured_max_frames setup
target_talk_setup = """        short_streaming = on_audio_chunk is not None and min_frames < ARIA_MIN_FRAMES
        spoken_units, numeric_extra_units, T_gen = _estimate_talker_frame_budget(
            text,
            n_words,
            min_frames,
            short_streaming=short_streaming,
        )
        text_derived_T_gen = T_gen
        diagnostic_max_frames = (
            quality_diagnostic.get("max_acoustic_frame_budget")
            if isinstance(quality_diagnostic, dict)
            else None
        )
        T_gen = apply_diagnostic_frame_budget(
            T_gen,
            diagnostic_max_frames=diagnostic_max_frames,
            absolute_max_frames=ARIA_MAX_FRAMES,
        )
        if max_frames is not None:
            configured_max_frames = int(max_frames)
            if configured_max_frames <= 0:
                raise ValueError("max_frames must be positive when provided")
            T_gen = min(T_gen, configured_max_frames)
            if metrics is not None:
                metrics["configured_max_frames"] = configured_max_frames
        else:
            configured_max_frames = ARIA_MAX_FRAMES
        live_frame_cap = min(ARIA_MAX_FRAMES, configured_max_frames)
        live_eos_frame_cap = live_frame_cap
        if diagnostic_max_frames is not None:
            live_eos_frame_cap = min(live_eos_frame_cap, int(diagnostic_max_frames))"""

replacement_talk_setup = """        short_streaming = on_audio_chunk is not None and min_frames < ARIA_MIN_FRAMES
        if max_frames is not None:
            configured_max_frames = int(max_frames)
            if configured_max_frames <= 0:
                raise ValueError("max_frames must be positive when provided")
            configured_max_frames = min(configured_max_frames, ARIA_HARD_MAX_FRAMES)
            if metrics is not None:
                metrics["configured_max_frames"] = configured_max_frames
        else:
            configured_max_frames = ARIA_MAX_FRAMES
        effective_absolute_max_frames = min(
            ARIA_HARD_MAX_FRAMES,
            max(ARIA_MAX_FRAMES, configured_max_frames),
        )
        spoken_units, numeric_extra_units, T_gen = _estimate_talker_frame_budget(
            text,
            n_words,
            min_frames,
            short_streaming=short_streaming,
            max_frames=configured_max_frames,
        )
        text_derived_T_gen = T_gen
        diagnostic_max_frames = (
            quality_diagnostic.get("max_acoustic_frame_budget")
            if isinstance(quality_diagnostic, dict)
            else None
        )
        T_gen = apply_diagnostic_frame_budget(
            T_gen,
            diagnostic_max_frames=diagnostic_max_frames,
            absolute_max_frames=effective_absolute_max_frames,
        )
        T_gen = min(T_gen, configured_max_frames)
        live_frame_cap = configured_max_frames
        live_eos_frame_cap = live_frame_cap
        if diagnostic_max_frames is not None:
            live_eos_frame_cap = min(live_eos_frame_cap, int(diagnostic_max_frames))"""

if target_talk_setup not in content:
    target_talk_setup = target_talk_setup.replace("\n", "\r\n")
    replacement_talk_setup = replacement_talk_setup.replace("\n", "\r\n")
assert target_talk_setup in content, "target_talk_setup not found"
content = content.replace(target_talk_setup, replacement_talk_setup, 1)

# 4. Live text loop budget
target_live_loop = """                    live_spoken_units, live_numeric_extra, live_text_budget = (
                        _estimate_talker_frame_budget(
                            live_text,
                            live_words,
                            min_frames,
                            short_streaming=short_streaming,
                        )
                    )
                    # Each native text token needs one Q1 query after a CB0
                    # frame; reserve two additional frames to feed native text
                    # EOS and sample codec EOS after the tail drains.
                    live_text_budget = max(
                        live_text_budget,
                        live_qwen_text_stream.observed_qwen_token_count + 2,
                    )
                    live_text_budget = apply_diagnostic_frame_budget(
                        live_text_budget,
                        diagnostic_max_frames=diagnostic_max_frames,
                        absolute_max_frames=ARIA_MAX_FRAMES,
                    )"""

replacement_live_loop = """                    live_spoken_units, live_numeric_extra, live_text_budget = (
                        _estimate_talker_frame_budget(
                            live_text,
                            live_words,
                            min_frames,
                            short_streaming=short_streaming,
                            max_frames=configured_max_frames,
                        )
                    )
                    # Each native text token needs one Q1 query after a CB0
                    # frame; reserve two additional frames to feed native text
                    # EOS and sample codec EOS after the tail drains.
                    live_text_budget = max(
                        live_text_budget,
                        live_qwen_text_stream.observed_qwen_token_count + 2,
                    )
                    live_text_budget = apply_diagnostic_frame_budget(
                        live_text_budget,
                        diagnostic_max_frames=diagnostic_max_frames,
                        absolute_max_frames=effective_absolute_max_frames,
                    )"""

if target_live_loop not in content:
    target_live_loop = target_live_loop.replace("\n", "\r\n")
    replacement_live_loop = replacement_live_loop.replace("\n", "\r\n")
assert target_live_loop in content, "target_live_loop not found"
content = content.replace(target_live_loop, replacement_live_loop, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("duplex_kernel/engine.py patched successfully!")
