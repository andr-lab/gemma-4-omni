from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from duplex_kernel.future_prefix import FuturePrefixQwen, SpeakerStepOutput
from duplex_kernel.models.talker import TalkerModel
from duplex_kernel.subtalker_graph import SubtalkerGraph

logger = logging.getLogger("duplex_kernel.speaker_runtime")


@dataclass(frozen=True)
class SpeakerStepResult:
    """Result of a single 80 ms speaker frame execution."""

    frame_id: int
    cb0_tensor: torch.Tensor
    rvq16_tensor: torch.Tensor
    c0_embed: torch.Tensor
    last_hidden: torch.Tensor
    visible_prefix_end: int
    event: torch.cuda.Event | None = None


class SpeakerRuntime:
    """Unified production speaker runtime managing Qwen talker and SubTalker graphs.

    All token selection and tensor operations remain GPU-resident to eliminate
    host PCIe round-trips and CPU blocking.
    """

    def __init__(
        self,
        speaker: FuturePrefixQwen,
        subtalker_graph: SubtalkerGraph,
        talker: TalkerModel,
        *,
        device: str = "cuda:0",
    ) -> None:
        self.speaker = speaker
        self.subtalker_graph = subtalker_graph
        self.talker = talker
        self.device = device
        self._current_frame_id = 0

    @property
    def current_frame_id(self) -> int:
        return self._current_frame_id

    def begin(self, text: str, voice_conditioning: dict[str, Any] | None = None) -> None:
        """Initialize a new utterance turn."""
        self.speaker.begin(text, voice_conditioning=voice_conditioning)
        self._current_frame_id = 0

    def push_v9_frame(self, frame_id: int, conditioning_frame: torch.Tensor) -> None:
        """Push causal V9 conditioning vector (1, 2048) or (2048,)."""
        self.speaker.push_v9_frame(frame_id, conditioning_frame)

    def step(
        self,
        conditioning_frame: torch.Tensor | None = None,
        *,
        frame_id: int | None = None,
        record_event: bool = True,
        do_sample: bool = True,
    ) -> SpeakerStepResult:
        """Execute one 80 ms speaker tick entirely on GPU.

        1. Pushes conditioning vector (if provided).
        2. Evaluates Qwen slot-KV forward step.
        3. Selects CB0 token directly in VRAM.
        4. Replays 15 SubTalker CUDA graphs for CB1..15.
        5. Assembles codes_16 on CUDA and commits to speaker state.
        6. Records optional CUDA event for asynchronous synchronization.
        """
        fid = self._current_frame_id if frame_id is None else int(frame_id)

        if conditioning_frame is not None:
            self.speaker.push_v9_frame(fid, conditioning_frame)

        # 1. Forward Qwen talker backbone
        step_out: SpeakerStepOutput = self.speaker.step_frame()

        # 2. Select CB0 token on GPU
        # In greedy or sampling mode, keep selection entirely on CUDA.
        # Note: Code2Wav RVQ Codebook 0 has strictly 2048 entries.
        cb0_logits = step_out.logits[:, :1, :2048]
        if not do_sample:
            c0_tensor = cb0_logits.argmax(dim=-1).squeeze().view(1)
        else:
            # TopK/TopP GPU sampling on CB0 logits (strictly in [0, 2047])
            topk_vals, topk_idx = torch.topk(cb0_logits, k=20, dim=-1)
            probs = torch.softmax(topk_vals / 0.80, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask = (cum - probs) > 0.90
            probs[mask] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            sampled_idx = torch.multinomial(probs.view(1, -1), num_samples=1)
            c0_tensor = topk_idx.view(1, -1).gather(1, sampled_idx).squeeze().view(1)
        c0_tensor = torch.clamp(c0_tensor, 0, 2047)

        # 3. GPU embedding projection
        c0_embed = self.talker.talker.get_input_embeddings()(c0_tensor.view(1, 1))

        # 4. Replay 15 SubTalker CUDA graphs
        sub_codes = self.subtalker_graph.generate(
            step_out.last_hidden,
            c0_token=c0_tensor,
            c0_embed=c0_embed,
            do_sample=do_sample,
            return_tensor=True,
        )

        # 5. Assemble codes_16 on GPU and commit
        if not isinstance(sub_codes, torch.Tensor):
            sub_codes = torch.tensor(sub_codes, device=self.device, dtype=torch.long)
        codes_16 = torch.cat([c0_tensor.view(1), sub_codes.view(-1)], dim=0)
        codes_16 = torch.clamp(codes_16, 0, 2047)
        self.speaker.commit_codes(codes_16)

        # 6. Record CUDA event
        event: torch.cuda.Event | None = None
        if record_event and torch.cuda.is_available():
            event = torch.cuda.Event(enable_timing=True)
            event.record()

        self._current_frame_id = fid + 1

        return SpeakerStepResult(
            frame_id=fid,
            cb0_tensor=c0_tensor,
            rvq16_tensor=codes_16,
            c0_embed=c0_embed,
            last_hidden=step_out.last_hidden,
            visible_prefix_end=step_out.visible_prefix_end,
            event=event,
        )

    def decode_frame(self, rvq16_tensor: torch.Tensor) -> np.ndarray:
        """Decode single 16-code frame to 24 kHz audio (1920 samples)."""
        if self.talker.code2wav is None:
            raise RuntimeError("Code2Wav non presente nel TalkerModel")
        codes = rvq16_tensor.view(1, 1, 16)
        with torch.inference_mode():
            out = self.talker.code2wav.decode(codes)
            audio = out.audio_values[0].detach().cpu().numpy().astype(np.float32)
        return audio


__all__ = [
    "SpeakerStepResult",
    "SpeakerRuntime",
]
