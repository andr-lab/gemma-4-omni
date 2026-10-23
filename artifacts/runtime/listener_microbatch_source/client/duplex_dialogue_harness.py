# -*- coding: utf-8 -*-
"""client/duplex_dialogue_harness.py

Reproducible full-duplex spoken dialogue harness for two interlocutors A and B.

Key Capabilities:
1. Single server process, two isolated WebSocket sessions (/v1/realtime?duplex=1)
   sharing a single model instance.
2. Cross-session isolation: independent KV cache, talker state, generations,
   voices, and histories.
3. Audio timeline reconstruction with packet ID tracking, deduplication, and
   rejection of cancelled/stale packets.
4. Polyphase 24 kHz -> 16 kHz resampling (exact 2/3 ratio) and 80 ms (1280 samples)
   pacing in real time.
5. Strict anti-echo routing: A's output routes solely to B's input; B's output routes
   solely to A's input.
6. Guardrails: max turns, max duration, silence timeouts, loop prevention.
7. Multi-track audio recording: separate WAV files for A-in, A-out, B-in, B-out.
8. Event-level JSONL tracing with microsecond monotonic timestamps.
9. Mode (a): human pre-recorded/transcribed audio playback with interruptions.
10. Mode (b): autonomous A <-> B conversation initiated from a scenario.
"""

from __future__ import annotations

import asyncio
import base64
import collections
from dataclasses import dataclass
import io
import json
import logging
import math
from pathlib import Path
import queue
import threading
import time
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import wave

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger("duplex_dialogue_harness")

SAMPLE_RATE_INPUT = 16_000   # Mic / ingress rate: 16 kHz
SAMPLE_RATE_OUTPUT = 24_000  # Server / Code2Wav output rate: 24 kHz
SCENARIO_AUDIO_DIR = Path("artifacts/scenario_audio_qwen3tts")
MODE_B_KICKOFF_FILENAME = "mode_b_kickoff.wav"
FRAME_DURATION_MS = 80
FRAME_SAMPLES_16K = SAMPLE_RATE_INPUT * FRAME_DURATION_MS // 1000  # 1280 samples
FRAME_SAMPLES_24K = SAMPLE_RATE_OUTPUT * FRAME_DURATION_MS // 1000  # 1920 samples


def _conditioning_ack_matches_frame(message: Dict[str, Any], expected_frame_id: int) -> bool:
    """Validate the listener's conditioning publication for one ingress frame."""
    try:
        return (
            int(message.get("frame_id", -1)) == expected_frame_id
            and int(message.get("conditioning_frame_id", -1)) == expected_frame_id
            and int(message.get("listener_output_frame_id", -1)) == expected_frame_id
            and message.get("conditioning_frame_available") is True
        )
    except (OverflowError, TypeError, ValueError):
        return False


def read_scenario_audio_wav(path: Path) -> np.ndarray:
    """Read a mono 16 kHz PCM16 scenario WAV as normalized float32 audio."""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != SAMPLE_RATE_INPUT or wf.getsampwidth() != 2:
            raise ValueError(
                f"Scenario audio must be mono 16 kHz PCM16: {path} "
                f"(channels={wf.getnchannels()}, rate={wf.getframerate()}, width={wf.getsampwidth()})"
            )
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32767.0
    if pcm.size == 0 or not np.isfinite(pcm).all():
        raise ValueError(f"Scenario audio is empty or non-finite: {path}")
    return np.clip(pcm, -1.0, 1.0).astype(np.float32, copy=False)


def resample_24k_to_16k(pcm_24k: np.ndarray) -> np.ndarray:
    """Resample 24 kHz float32 audio to 16 kHz float32 using exact 2/3 polyphase FIR filter."""
    pcm = np.asarray(pcm_24k, dtype=np.float32).reshape(-1)
    if pcm.size == 0:
        return np.empty(0, dtype=np.float32)

    if HAS_SCIPY:
        # 24000 * 2 / 3 = 16000 exact
        resampled = scipy.signal.resample_poly(pcm, up=2, down=3).astype(np.float32)
    else:
        # Fallback linear interpolation
        old_indices = np.arange(pcm.size)
        new_length = int(round(pcm.size * 2.0 / 3.0))
        new_indices = np.linspace(0, pcm.size - 1, new_length)
        resampled = np.interp(new_indices, old_indices, pcm).astype(np.float32)

    return resampled


def pcm_to_80ms_frames(
    pcm_16k: np.ndarray,
    remainder: np.ndarray | None = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Slice 16 kHz audio into exact 80 ms (1280 sample) frames, returning frames and leftover remainder."""
    if remainder is not None and remainder.size > 0:
        combined = np.concatenate((remainder, pcm_16k))
    else:
        combined = np.asarray(pcm_16k, dtype=np.float32)

    num_full = combined.size // FRAME_SAMPLES_16K
    frames: List[np.ndarray] = []
    for i in range(num_full):
        start = i * FRAME_SAMPLES_16K
        end = start + FRAME_SAMPLES_16K
        frames.append(combined[start:end])

    rem = combined[num_full * FRAME_SAMPLES_16K:]
    return frames, rem


def float_to_pcm16_bytes(pcm_float: np.ndarray) -> bytes:
    """Convert float32 [-1.0, 1.0] samples to signed 16-bit little-endian bytes."""
    clipped = np.clip(pcm_float, -1.0, 1.0)
    int16_arr = (clipped * 32767.0).astype("<i2")
    return int16_arr.tobytes()


def pcm16_bytes_to_float(pcm_bytes: bytes) -> np.ndarray:
    """Convert signed 16-bit little-endian bytes to float32 [-1.0, 1.0]."""
    int16_arr = np.frombuffer(pcm_bytes, dtype="<i2")
    return int16_arr.astype(np.float32) / 32767.0


class AudioTimelineReconstructor:
    """Reconstructs the continuous timeline for server-emitted audio packets.

    Filters out duplicates and stale packets from cancelled generations.
    """

    def __init__(self) -> None:
        self.seen_packets: Set[Tuple[str, int]] = set()
        self.cancelled_responses: Set[str] = set()
        self.packets_by_response: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        self.duplicate_packet_count = 0
        self.cancelled_packet_count = 0
        self._lock = threading.Lock()

    def cancel_response(self, response_id: str) -> None:
        with self._lock:
            self.cancelled_responses.add(response_id)

    def add_packet(
        self,
        *,
        response_id: str,
        generation_id: int,
        packet_id: int,
        first_frame_id: int,
        last_frame_id: int,
        start_sample: int,
        end_sample: int,
        pcm: np.ndarray,
    ) -> bool:
        with self._lock:
            if response_id in self.cancelled_responses:
                self.cancelled_packet_count += 1
                return False

            key = (response_id, int(packet_id))
            if key in self.seen_packets:
                self.duplicate_packet_count += 1
                return False

            self.seen_packets.add(key)
            self.packets_by_response[response_id].append({
                "packet_id": int(packet_id),
                "generation_id": int(generation_id),
                "first_frame_id": int(first_frame_id),
                "last_frame_id": int(last_frame_id),
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "pcm": np.asarray(pcm, dtype=np.float32).copy(),
            })
            return True

    def get_coherent_pcm(self, response_id: Optional[str] = None) -> np.ndarray:
        with self._lock:
            chunks: List[np.ndarray] = []
            if response_id is not None:
                pkts = sorted(self.packets_by_response.get(response_id, []), key=lambda p: p["packet_id"])
                for p in pkts:
                    chunks.append(p["pcm"])
            else:
                for resp_id, pkts in self.packets_by_response.items():
                    if resp_id not in self.cancelled_responses:
                        for p in sorted(pkts, key=lambda x: x["packet_id"]):
                            chunks.append(p["pcm"])

            if not chunks:
                return np.empty(0, dtype=np.float32)
            return np.concatenate(chunks)


class AudioBridge:
    """Bounded, anti-echo 80 ms bridge between A and B."""

    def __init__(self, *, max_pending_frames: int = 1024) -> None:
        if max_pending_frames < 1:
            raise ValueError("max_pending_frames deve essere positivo")
        self.max_pending_frames = int(max_pending_frames)
        self._queue_a_to_b: List[np.ndarray] = []
        self._queue_b_to_a: List[np.ndarray] = []
        self._remainder_a: np.ndarray = np.empty(0, dtype=np.float32)
        self._remainder_b: np.ndarray = np.empty(0, dtype=np.float32)
        self._lock = threading.Lock()

    def feed_output(self, source_speaker: str, pcm_24k: np.ndarray) -> None:
        """Route 24 kHz output from source_speaker strictly to the opposing interlocutor."""
        if pcm_24k.size == 0:
            return

        pcm_16k = resample_24k_to_16k(pcm_24k)

        with self._lock:
            if source_speaker.upper() == "A":
                target_queue, remainder = self._queue_a_to_b, self._remainder_a
            elif source_speaker.upper() == "B":
                target_queue, remainder = self._queue_b_to_a, self._remainder_b
            else:
                raise ValueError(f"Unknown source speaker: {source_speaker}")
            frames, next_remainder = pcm_to_80ms_frames(pcm_16k, remainder)
            if len(target_queue) + len(frames) > self.max_pending_frames:
                raise OverflowError(
                    f"AudioBridge {source_speaker.upper()} capacity {self.max_pending_frames} frames exceeded"
                )
            target_queue.extend(frames)
            if source_speaker.upper() == "A":
                self._remainder_a = next_remainder
            else:
                self._remainder_b = next_remainder

    def get_pending_ingress_frames(self, target_speaker: str) -> List[np.ndarray]:
        """Fetch all queued 80 ms frames for target_speaker without echo."""
        with self._lock:
            if target_speaker.upper() == "B":
                frames = list(self._queue_a_to_b)
                self._queue_a_to_b.clear()
                return frames
            elif target_speaker.upper() == "A":
                frames = list(self._queue_b_to_a)
                self._queue_b_to_a.clear()
                return frames
            else:
                raise ValueError(f"Unknown target speaker: {target_speaker}")

    def pop_single_frame(self, target_speaker: str) -> Optional[np.ndarray]:
        """Fetch the next queued 80 ms frame for target_speaker without echo."""
        with self._lock:
            if target_speaker.upper() == "B":
                if self._queue_a_to_b:
                    return self._queue_a_to_b.pop(0)
                return None
            elif target_speaker.upper() == "A":
                if self._queue_b_to_a:
                    return self._queue_b_to_a.pop(0)
                return None
            return None

    def pending_count(self, target_speaker: str) -> int:
        """Count pending queued 80 ms frames for target_speaker."""
        with self._lock:
            if target_speaker.upper() == "B":
                return len(self._queue_a_to_b)
            elif target_speaker.upper() == "A":
                return len(self._queue_b_to_a)
            return 0

    def finalize_response(self, source_speaker: str) -> int:
        """Pad and enqueue a response's final partial 80 ms frame, if any."""
        source = source_speaker.upper()
        with self._lock:
            if source == "A":
                target_queue, remainder = self._queue_a_to_b, self._remainder_a
            elif source == "B":
                target_queue, remainder = self._queue_b_to_a, self._remainder_b
            else:
                raise ValueError(f"Unknown source speaker: {source_speaker}")
            if not remainder.size:
                return 0
            if len(target_queue) >= self.max_pending_frames:
                raise OverflowError(f"AudioBridge {source} capacity {self.max_pending_frames} frames exceeded")
            padded = np.zeros(FRAME_SAMPLES_16K, dtype=np.float32)
            padded[:remainder.size] = remainder
            target_queue.append(padded)
            if source == "A":
                self._remainder_a = np.empty(0, dtype=np.float32)
            else:
                self._remainder_b = np.empty(0, dtype=np.float32)
            return 1

    def clear_channel(self, target_speaker: str) -> Dict[str, int]:
        """Discard cancelled audio and report the exact dropped frame/tail counts."""
        with self._lock:
            if target_speaker.upper() == "B":
                discarded = {"frames": len(self._queue_a_to_b), "remainder_samples": int(self._remainder_a.size)}
                self._queue_a_to_b.clear()
                self._remainder_a = np.empty(0, dtype=np.float32)
                return discarded
            if target_speaker.upper() == "A":
                discarded = {"frames": len(self._queue_b_to_a), "remainder_samples": int(self._remainder_b.size)}
                self._queue_b_to_a.clear()
                self._remainder_b = np.empty(0, dtype=np.float32)
                return discarded
            raise ValueError(f"Unknown target speaker: {target_speaker}")


class DialogueRecorder:
    """Records separate WAV audio tracks for A-in, A-out, B-in, B-out."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracks: Dict[str, List[np.ndarray]] = {
            "interlocutor_a_input": [],   # 16 kHz
            "interlocutor_a_output": [],  # 24 kHz
            "interlocutor_b_input": [],   # 16 kHz
            "interlocutor_b_output": [],  # 24 kHz
        }
        self._lock = threading.Lock()

    def record_input(self, speaker: str, pcm_16k: np.ndarray) -> None:
        with self._lock:
            key = f"interlocutor_{speaker.lower()}_input"
            self.tracks[key].append(np.asarray(pcm_16k, dtype=np.float32).copy())

    def record_output(self, speaker: str, pcm_24k: np.ndarray) -> None:
        with self._lock:
            key = f"interlocutor_{speaker.lower()}_output"
            self.tracks[key].append(np.asarray(pcm_24k, dtype=np.float32).copy())

    def finalize(self) -> Dict[str, Path]:
        """Save all audio tracks to standard mono 16-bit PCM WAV files."""
        with self._lock:
            saved: Dict[str, Path] = {}
            for name, chunks in self.tracks.items():
                sample_rate = SAMPLE_RATE_OUTPUT if "output" in name else SAMPLE_RATE_INPUT
                path = self.output_dir / f"{name}.wav"
                if chunks:
                    full_pcm = np.concatenate(chunks)
                else:
                    full_pcm = np.empty(0, dtype=np.float32)

                pcm16_bytes = float_to_pcm16_bytes(full_pcm)
                with wave.open(str(path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm16_bytes)
                saved[name] = path
            return saved


class DialogueEventLogger:
    """Thread-safe event logger writing JSONL with structured metadata."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", encoding="utf-8")
        self._lock = threading.Lock()

    def log(
        self,
        event_type: str,
        *,
        speaker: str = "",
        turn_index: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime()) + f"{int(time.time() * 1000) % 1000:03d}Z",
            "timestamp_ns": time.monotonic_ns(),
            "event_type": str(event_type),
            "speaker": str(speaker),
            "turn_index": int(turn_index),
            "details": details or {},
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


class DialogueConfig:
    def __init__(
        self,
        *,
        max_turns: int = 4,
        max_duration_s: float = 60.0,
        silence_timeout_s: float = 10.0,
        frame_pacing_s: float = 0.080,
        ingress_ack_timeout_s: float = 2.0,
        ws_url: str = "ws://localhost:8000/v1/realtime?duplex=1",
        voice_a: str = "male_it",
        voice_b: str = "female_it",
        streaming_voice_profile: str = "gemma",
        streaming_turn_policy: str = "gemma_native_control6",
        turn_policy: str = "none",
        instructions_a: str = "Sei Marco, un cliente che interagisce in modo chiaro e naturale.",
        instructions_b: str = "Sei un assistente AI accogliente e professionale. Rispondi in modo conciso.",
    ) -> None:
        self.max_turns = max_turns
        self.max_duration_s = max_duration_s
        self.silence_timeout_s = silence_timeout_s
        if ingress_ack_timeout_s <= 0.0:
            raise ValueError("ingress_ack_timeout_s deve essere positivo")
        self.frame_pacing_s = frame_pacing_s
        self.ingress_ack_timeout_s = ingress_ack_timeout_s
        self.ws_url = ws_url
        self.voice_a = voice_a
        self.voice_b = voice_b
        self.streaming_voice_profile = str(streaming_voice_profile).strip()
        if not self.streaming_voice_profile:
            raise ValueError("streaming_voice_profile deve essere non vuoto")
        self.streaming_turn_policy = str(streaming_turn_policy).strip()
        if not self.streaming_turn_policy:
            raise ValueError("streaming_turn_policy deve essere non vuoto")
        self.turn_policy = turn_policy
        self.instructions_a = instructions_a
        self.instructions_b = instructions_b


@dataclass
class DecodedAudioPacket:
    response_id: str
    generation_id: int
    packet_id: int
    first_frame_id: int
    last_frame_id: int
    start_sample: int
    end_sample: int
    sample_rate: int
    pcm: np.ndarray


@dataclass(frozen=True)
class StreamingDialogueStep:
    """One user audio phrase in a concurrently streamed single-session scenario."""

    step_id: str
    pcm_16k: np.ndarray
    expected_text: str
    start_after_s: float = 0.0
    after_first_consumed_s: Optional[float] = None
    commit_after: bool = True
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.step_id:
            raise ValueError("step_id non può essere vuoto")
        if self.start_after_s < 0 or (
            self.after_first_consumed_s is not None and self.after_first_consumed_s < 0
        ):
            raise ValueError("gli offset temporali dello scenario devono essere non negativi")
        if self.start_after_s and self.after_first_consumed_s is not None:
            raise ValueError("usare un solo riferimento temporale per ogni step")
        pcm = np.asarray(self.pcm_16k, dtype=np.float32).reshape(-1)
        if pcm.size == 0 or not np.isfinite(pcm).all():
            raise ValueError(f"PCM non valido nello step {self.step_id}")
        if np.any(np.abs(pcm) > 1.0):
            raise ValueError(f"PCM fuori [-1, 1] nello step {self.step_id}")
        object.__setattr__(self, "pcm_16k", pcm)


@dataclass(frozen=True)
class StreamingDialogueScenario:
    """Reusable single-user-session stream with phrase-boundary commits."""

    scenario_id: str
    system_prompt: str
    steps: List[StreamingDialogueStep]
    auto_response: bool = True
    turn_policy: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.steps:
            raise ValueError("scenario_id e almeno uno step sono richiesti")
        step_ids = [step.step_id for step in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("step_id duplicati nello scenario")


def streaming_scenario_from_fixtures(
    name: str,
    fixture_dir: Path = Path("artifacts/turn_local_rollback_harness_20261018/inputs_16k"),
) -> StreamingDialogueScenario:
    """Load the reusable A correction or B long-utterance scenario fixtures."""
    normalized = str(name).strip().upper()
    fixture_dir = Path(fixture_dir)
    if normalized == "A":
        initial_path = fixture_dir / "A_initial_milano.wav"
        correction_path = fixture_dir / "A_correction_torino.wav"
        return StreamingDialogueScenario(
            scenario_id="A_milano_correction_torino",
            system_prompt="Sei un assistente AI accogliente. Rispondi in italiano.",
            steps=[
                StreamingDialogueStep(
                    step_id="initial_milano",
                    pcm_16k=read_scenario_audio_wav(initial_path),
                    expected_text="Consigliami un itinerario di tre giorni a Milano",
                    source_path=str(initial_path.resolve()),
                ),
                StreamingDialogueStep(
                    step_id="correction_torino",
                    pcm_16k=read_scenario_audio_wav(correction_path),
                    expected_text="Scusa, intendevo dire Torino, non Milano",
                    after_first_consumed_s=1.5,
                    source_path=str(correction_path.resolve()),
                ),
            ],
        )
    if normalized == "B":
        utterance_path = fixture_dir / "B_docker_15s_pause300.wav"
        return StreamingDialogueScenario(
            scenario_id="B_docker_long_utterance",
            system_prompt="Sei un assistente AI accogliente. Rispondi in italiano.",
            steps=[
                StreamingDialogueStep(
                    step_id="docker_15s_with_pause",
                    pcm_16k=read_scenario_audio_wav(utterance_path),
                    expected_text=(
                        "Sto avviando il servizio. Docker è il software per i container. "
                        "Uso Python e CUDA."
                    ),
                    source_path=str(utterance_path.resolve()),
                ),
            ],
        )
    raise ValueError(f"Scenario streaming sconosciuto: {name!r}; usare A o B")


def validate_scenario_initial_sample_count(
    scenario: StreamingDialogueScenario,
    expected_sample_count: Optional[int],
) -> None:
    """Fail before connecting when a run requires an exact initial WAV duration."""
    if expected_sample_count is None:
        return
    if expected_sample_count <= 0:
        raise ValueError("expected_initial_sample_count deve essere positivo")
    first = scenario.steps[0]
    actual = int(first.pcm_16k.size)
    if actual != expected_sample_count:
        source = first.source_path or "<scenario PCM>"
        raise ValueError(
            f"Initial scenario input {source} has {actual} samples at {SAMPLE_RATE_INPUT} Hz; "
            f"expected {expected_sample_count} samples at {SAMPLE_RATE_INPUT} Hz"
        )


def decode_response_audio(event: Dict[str, Any]) -> DecodedAudioPacket:
    """Decode server response.audio.delta base64 WAV payload."""
    if event.get("type") != "response.audio.delta":
        raise ValueError("Event is not response.audio.delta")
    encoded = event["delta"]
    wav_bytes = base64.b64decode(encoded)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        raw_pcm = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw_pcm, dtype="<i2").astype(np.float32) / 32767.0
    return DecodedAudioPacket(
        response_id=str(event.get("response_id", "")),
        generation_id=int(event.get("generation_id", 0)),
        packet_id=int(event.get("packet_id", 0)),
        first_frame_id=int(event.get("first_frame_id", 0)),
        last_frame_id=int(event.get("last_frame_id", 0)),
        start_sample=int(event.get("start_sample", 0)),
        end_sample=int(event.get("end_sample", 0)),
        sample_rate=int(event.get("sample_rate", SAMPLE_RATE_OUTPUT)),
        pcm=pcm,
    )


class DialogueOrchestrator:
    """Orchestrates multi-turn full-duplex spoken dialogue testing across Mode (a) and Mode (b)."""

    def __init__(
        self,
        config: Optional[DialogueConfig] = None,
        output_dir: Optional[Path] = None,
        ws_connect_fn: Optional[Callable] = None,
        scenario_audio_dir: Optional[Path] = None,
        asr_transcriber: Optional[Callable[[np.ndarray, int], str]] = None,
    ) -> None:
        from dialogue_eval.evaluator import DialogueEvaluator

        self.config = config or DialogueConfig()
        self.output_dir = Path(output_dir or "artifacts/dialogue_run")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scenario_audio_dir = Path(scenario_audio_dir or SCENARIO_AUDIO_DIR)
        self.asr_transcriber = asr_transcriber
        self.recorder = DialogueRecorder(self.output_dir)
        self.event_logger = DialogueEventLogger(self.output_dir / "dialogue_events.jsonl")
        self.evaluator = DialogueEvaluator()
        bridge_capacity = max(1, math.ceil(self.config.max_duration_s * SAMPLE_RATE_INPUT / FRAME_SAMPLES_16K))
        self.bridge = AudioBridge(max_pending_frames=bridge_capacity)
        self.timeline_a = AudioTimelineReconstructor()
        self.timeline_b = AudioTimelineReconstructor()
        self.ws_connect_fn = ws_connect_fn

    async def _transcribe_input_audio(self, pcm_16k: np.ndarray) -> Tuple[str, bool, Optional[str]]:
        """Transcribe the exact ingress PCM with an injected independent ASR backend."""
        if self.asr_transcriber is None or pcm_16k.size == 0:
            return "", False, None
        try:
            transcript = await asyncio.to_thread(
                self.asr_transcriber,
                np.asarray(pcm_16k, dtype=np.float32).copy(),
                SAMPLE_RATE_INPUT,
            )
            if not isinstance(transcript, str):
                raise TypeError(f"ASR returned {type(transcript).__name__}, expected str")
            return transcript.strip(), True, None
        except Exception as exc:
            return "", False, f"{type(exc).__name__}: {exc}"

    async def _connect_ws(self, url: str) -> Any:
        if self.ws_connect_fn is not None:
            return await self.ws_connect_fn(url)
        import websockets
        return await websockets.connect(
            url,
            max_size=1_000_000,
            ping_interval=20,
            ping_timeout=20,
        )

    async def _setup_session(
        self,
        ws: Any,
        speaker_name: str,
        voice: str,
        instructions: str,
        turn_policy: str = "turnhead",
        auto_response: bool = False,
        input_audio_buffer_acknowledgements: bool = False,
        input_audio_buffer_conditioning_acknowledgements: bool = False,
    ) -> Dict[str, Any]:
        raw_created = await ws.recv()
        created = json.loads(raw_created)
        self.event_logger.log("session.created", speaker=speaker_name, details=created)

        update_msg = {
            "type": "session.update",
            "session": {
                "voice": voice,
                "instructions": instructions,
                "auto_response": auto_response,
                "turn_policy": turn_policy,
                "input_audio_buffer_acknowledgements": input_audio_buffer_acknowledgements,
                "input_audio_buffer_conditioning_acknowledgements": (
                    input_audio_buffer_conditioning_acknowledgements
                ),
            },
        }
        await ws.send(json.dumps(update_msg))
        raw_updated = await ws.recv()
        updated = json.loads(raw_updated)
        self.event_logger.log(str(updated.get("type", "session.update.error")), speaker=speaker_name, details=updated)
        if updated.get("type") != "session.updated":
            raise RuntimeError(f"session update rejected for {speaker_name}: {updated!r}")
        await asyncio.sleep(0.1)
        return updated

    async def run_streaming_scenario(
        self,
        scenario: StreamingDialogueScenario,
    ) -> Dict[str, Any]:
        """Stream user phrases, assistant events, and logical playback on one WS session."""
        from client.openai_duplex_client import BoundedPCMPlayback, decode_response_audio as decode_playback_packet

        self.event_logger.log(
            "streaming_scenario_started",
            details={
                "scenario_id": scenario.scenario_id,
                "auto_response": scenario.auto_response,
                "turn_policy": scenario.turn_policy or self.config.streaming_turn_policy,
                "voice_profile": self.config.streaming_voice_profile,
                "frame_pacing_s": self.config.frame_pacing_s,
                "playback_consumption_semantics": "logical_client_device_buffer",
                "input_fixtures": [
                    {
                        "step_id": step.step_id,
                        "resolved_input_path": step.source_path,
                        "sample_rate": SAMPLE_RATE_INPUT,
                        "sample_count": int(step.pcm_16k.size),
                    }
                    for step in scenario.steps
                ],
            },
        )
        ws_ctx = await self._connect_ws(self.config.ws_url)
        result: Dict[str, Any] = {}
        async with ws_ctx as ws:
            await self._setup_session(
                ws,
                speaker_name="A",
                voice=self.config.streaming_voice_profile,
                instructions=scenario.system_prompt,
                turn_policy=scenario.turn_policy or self.config.streaming_turn_policy,
                auto_response=scenario.auto_response,
                input_audio_buffer_acknowledgements=True,
                input_audio_buffer_conditioning_acknowledgements=True,
            )

            playback = BoundedPCMPlayback()
            ack_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
            stop_tasks = asyncio.Event()
            producer_finished = asyncio.Event()
            first_pcm_consumed = asyncio.Event()
            frame_consumed_event = asyncio.Event()
            send_lock = asyncio.Lock()
            input_frame_lock = asyncio.Lock()
            event_lock = asyncio.Lock()
            next_input_frame_deadline = 0.0
            state: Dict[str, Any] = {
                "input_active": False,
                "input_frame_id": 0,
                "frames_consumed": 0,
                "ingress_acks_observed": False,
                "ingress_sample_cursor": 0,
                "playback_samples_consumed": 0,
                "active_responses": set(),
                "response_generations": {},
                "responses": {},
                "transcripts": {},
                "step_schedule": {},
                "policy_diagnostics": [],
                "consumed_acknowledgements": [],
                "commit_timestamps_ns": [],
                "pending_committed_steps": {},
                "response_pending_steps": {},
                "server_errors": [],
                "last_event_ns": time.monotonic_ns(),
                "last_commit_ns": None,
                "last_terminal_ns": None,
                "first_response_created_ns": None,
                "first_audio_delta_ns": None,
                "first_pcm_consumed_ns": None,
                "last_consumed_ns_by_response": {},
                "shadow_tracker_errors": [],
                "first_ack_sent_ns": None,
                "stop_ns": None,
                "stop_reason": "running",
                "producer_errors": [],
            }
            run_started_ns = time.monotonic_ns()
            duration_limit_s = max(0.01, float(self.config.max_duration_s))
            transcription_tasks: List[asyncio.Task] = []
            shadow_chunks: collections.deque[Dict[str, Any]] = collections.deque()
            shadow_active_response: Optional[str] = None
            shadow_completed_responses: Set[str] = set()
            shadow_generation_id: Optional[int] = None

            def shadow_drop(predicate: Callable[[Dict[str, Any]], bool]) -> None:
                kept = collections.deque()
                for chunk in shadow_chunks:
                    if not predicate(chunk):
                        kept.append(chunk)
                shadow_chunks.clear()
                shadow_chunks.extend(kept)

            def shadow_begin_response(response_id: str) -> None:
                nonlocal shadow_active_response
                if (
                    shadow_active_response is not None
                    and shadow_active_response != response_id
                    and shadow_active_response not in shadow_completed_responses
                ):
                    previous = shadow_active_response
                    shadow_drop(lambda chunk: chunk["response_id"] == previous)
                shadow_active_response = response_id
                shadow_completed_responses.discard(response_id)

            def shadow_advance_generation(generation_id: int) -> None:
                nonlocal shadow_generation_id
                if shadow_generation_id is not None and generation_id < shadow_generation_id:
                    return
                if shadow_generation_id is None or generation_id > shadow_generation_id:
                    shadow_drop(
                        lambda chunk: (
                            chunk["generation_id"] < generation_id
                            and chunk["response_id"] not in shadow_completed_responses
                        )
                    )
                    shadow_generation_id = generation_id

            def shadow_cancel_response(response_id: Optional[str], generation_id: Optional[int]) -> None:
                nonlocal shadow_active_response
                if response_id is None:
                    shadow_chunks.clear()
                    shadow_completed_responses.clear()
                    shadow_active_response = None
                    return
                shadow_drop(
                    lambda chunk: chunk["response_id"] == response_id
                    and (generation_id is None or chunk["generation_id"] == generation_id)
                )
                shadow_completed_responses.discard(response_id)
                remaining_generations = state["response_generations"].get(response_id, set())
                if shadow_active_response == response_id and (generation_id is None or not remaining_generations):
                    shadow_active_response = None

            def shadow_consume(sample_count: int, consumed_ns: int) -> List[Dict[str, Any]]:
                remaining = int(sample_count)
                portions: List[Dict[str, Any]] = []
                while remaining > 0 and shadow_chunks:
                    chunk = shadow_chunks[0]
                    available = int(chunk["end_sample"] - chunk["start_sample"] - chunk["offset"])
                    count = min(remaining, available)
                    start_sample = int(chunk["start_sample"] + chunk["offset"])
                    end_sample = start_sample + count
                    chunk["offset"] += count
                    remaining -= count
                    state["last_consumed_ns_by_response"][chunk["response_id"]] = consumed_ns
                    portions.append({
                        "response_id": chunk["response_id"],
                        "generation_id": chunk["generation_id"],
                        "speaker_segment_id": chunk["speaker_segment_id"],
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "sample_rate": chunk["sample_rate"],
                    })
                    if chunk["offset"] == chunk["end_sample"] - chunk["start_sample"]:
                        shadow_chunks.popleft()
                if remaining:
                    state["shadow_tracker_errors"].append(
                        f"playback copied {sample_count} samples without {remaining} samples of segment metadata"
                    )
                return portions

            async def send_input_frame(
                frame: np.ndarray,
                *,
                source: str,
                step_id: str = "",
                source_sample_start: Optional[int] = None,
                source_sample_end: Optional[int] = None,
                padded_samples: int = 0,
                idle_only: bool = False,
            ) -> bool:
                nonlocal next_input_frame_deadline
                async with input_frame_lock:
                    if idle_only and (state["input_active"] or stop_tasks.is_set()):
                        return False
                    pacing_delay = next_input_frame_deadline - time.monotonic()
                    if pacing_delay > 0.0:
                        await asyncio.sleep(pacing_delay)
                    if idle_only and (state["input_active"] or stop_tasks.is_set()):
                        return False

                    if state.get("ingress_acks_observed", False):
                        while (
                            state["input_frame_id"] - state.get("frames_consumed", 0) >= 3
                            and not stop_tasks.is_set()
                        ):
                            frame_consumed_event.clear()
                            try:
                                await asyncio.wait_for(frame_consumed_event.wait(), timeout=0.1)
                            except asyncio.TimeoutError:
                                break
                            if idle_only and (state["input_active"] or stop_tasks.is_set()):
                                return False
                    if idle_only and (state["input_active"] or stop_tasks.is_set()):
                        return False

                    message = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(float_to_pcm16_bytes(frame)).decode("ascii"),
                    }
                    step_timing = state["step_schedule"].get(step_id, {})
                    is_correction_first_sample = (
                        source == "scenario_audio"
                        and source_sample_start == 0
                        and step_timing.get("reference") == "first_client_pcm_consumed"
                    )
                    async with send_lock:
                        # Recheck after every potentially blocking wait. Idle silence
                        # must never take a frame ID after a WAV producer has resumed.
                        if idle_only and (state["input_active"] or stop_tasks.is_set()):
                            return False
                        frame_id = int(state["input_frame_id"])
                        ingress_start = int(state["ingress_sample_cursor"])
                        ingress_end = ingress_start + FRAME_SAMPLES_16K
                        if is_correction_first_sample:
                            # Snapshot the old playback tail and put the correction frame
                            # on the wire under the same event fence. A cancel cannot
                            # make the snapshot stale between those two operations.
                            async with event_lock:
                                if idle_only and (state["input_active"] or stop_tasks.is_set()):
                                    return False
                                step_timing["old_response_ids_at_first_correction_sample"] = sorted(
                                    {chunk["response_id"] for chunk in shadow_chunks}
                                )
                                await ws.send(json.dumps(message))
                                sent_ns = time.monotonic_ns()
                                step_timing["first_sample_sent_ns"] = sent_ns
                        else:
                            await ws.send(json.dumps(message))
                            sent_ns = time.monotonic_ns()
                        state["input_frame_id"] = frame_id + 1
                        state["ingress_sample_cursor"] = ingress_end
                        next_input_frame_deadline = sent_ns / 1e9 + max(
                            0.0, float(self.config.frame_pacing_s)
                        )
                        self.event_logger.log(
                            "raw_ws_send",
                            speaker="user",
                            turn_index=frame_id,
                            details=message,
                        )
                        self.recorder.record_input("A", frame)
                        self.event_logger.log(
                            "input_audio_frame_sent",
                            speaker="user",
                            turn_index=frame_id,
                            details={
                                "ingress_frame_id": frame_id,
                                "source": source,
                                "step_id": step_id,
                                "sent_ns": sent_ns,
                                "sample_rate": SAMPLE_RATE_INPUT,
                                "sample_count": FRAME_SAMPLES_16K,
                                "ingress_start_sample": ingress_start,
                                "ingress_end_sample": ingress_end,
                                "source_start_sample": source_sample_start,
                                "source_end_sample": source_sample_end,
                                "padded_samples": int(padded_samples),
                            },
                        )
                    return True

            async def record_asr(step: StreamingDialogueStep) -> None:
                transcript, available, error = await self._transcribe_input_audio(step.pcm_16k)
                state["transcripts"][step.step_id] = {
                    "expected_text": step.expected_text,
                    "asr_transcript": transcript,
                    "asr_available": available,
                    "sample_rate": SAMPLE_RATE_INPUT,
                    "sample_count": int(step.pcm_16k.size),
                    "source_sample_start": 0,
                    "source_sample_end": int(step.pcm_16k.size),
                    **({"asr_error": error} if error else {}),
                }
                self.event_logger.log(
                    "input_audio_transcribed",
                    speaker="user",
                    details={"step_id": step.step_id, **state["transcripts"][step.step_id]},
                )

            async def input_producer() -> None:
                try:
                    for step in scenario.steps:
                        if step.after_first_consumed_s is not None:
                            try:
                                await asyncio.wait_for(
                                    first_pcm_consumed.wait(),
                                    timeout=max(0.01, duration_limit_s - (time.monotonic_ns() - run_started_ns) / 1e9),
                                )
                            except asyncio.TimeoutError:
                                state["stop_reason"] = "first_pcm_timeout"
                                return
                            anchor_ns = int(state["first_pcm_consumed_ns"] or time.monotonic_ns())
                            target_ns = anchor_ns + int(step.after_first_consumed_s * 1e9)
                            await asyncio.sleep(max(0.0, (target_ns - time.monotonic_ns()) / 1e9))
                            schedule = {
                                "reference": "first_client_pcm_consumed",
                                "target_offset_s": step.after_first_consumed_s,
                                "anchor_ns": anchor_ns,
                                "target_ns": target_ns,
                            }
                        elif step.start_after_s > 0:
                            target_ns = run_started_ns + int(step.start_after_s * 1e9)
                            await asyncio.sleep(max(0.0, (target_ns - time.monotonic_ns()) / 1e9))
                            schedule = {
                                "reference": "scenario_start",
                                "target_offset_s": step.start_after_s,
                                "target_ns": target_ns,
                            }
                        else:
                            schedule = {"reference": "scenario_start", "target_offset_s": 0.0}

                        step_start_ns = time.monotonic_ns()
                        state["step_schedule"][step.step_id] = {
                            **schedule,
                            "actual_start_ns": step_start_ns,
                        }
                        self.event_logger.log(
                            "streaming_input_step_started",
                            speaker="user",
                            details={
                                "step_id": step.step_id,
                                "expected_text": step.expected_text,
                                "source_sample_count": int(step.pcm_16k.size),
                                **schedule,
                            },
                        )
                        async with input_frame_lock:
                            state["input_active"] = True
                        total_frames = math.ceil(step.pcm_16k.size / FRAME_SAMPLES_16K)
                        for frame_index in range(total_frames):
                            source_start = frame_index * FRAME_SAMPLES_16K
                            source_end = min(source_start + FRAME_SAMPLES_16K, step.pcm_16k.size)
                            source_chunk = step.pcm_16k[source_start:source_end]
                            frame = np.zeros(FRAME_SAMPLES_16K, dtype=np.float32)
                            frame[:source_chunk.size] = source_chunk
                            await send_input_frame(
                                frame,
                                source="scenario_audio",
                                step_id=step.step_id,
                                source_sample_start=source_start,
                                source_sample_end=source_end,
                                padded_samples=FRAME_SAMPLES_16K - int(source_chunk.size),
                            )

                        if step.commit_after:
                            async with send_lock:
                                commit_message = {"type": "input_audio_buffer.commit"}
                                await ws.send(json.dumps(commit_message))
                                commit_ns = time.monotonic_ns()
                                self.event_logger.log(
                                    "raw_ws_send",
                                    speaker="user",
                                    details=commit_message,
                                )
                            state["last_commit_ns"] = commit_ns
                            state["commit_timestamps_ns"].append(commit_ns)
                            state["pending_committed_steps"][step.step_id] = commit_ns
                            state["step_schedule"][step.step_id]["commit_ns"] = commit_ns
                            self.event_logger.log(
                                "input_audio_buffer_commit_sent",
                                speaker="user",
                                details={
                                    "step_id": step.step_id,
                                    "timestamp_ns": commit_ns,
                                    "phrase_sample_count": int(step.pcm_16k.size),
                                    "ingress_frame_count": int(state["input_frame_id"]),
                                },
                            )
                        async with input_frame_lock:
                            state["input_active"] = False
                        transcription_tasks.append(asyncio.create_task(record_asr(step)))
                    producer_finished.set()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    state["producer_errors"].append(f"{type(exc).__name__}: {exc}")
                    state["stop_reason"] = "input_error"
                finally:
                    async with input_frame_lock:
                        state["input_active"] = False
                    producer_finished.set()

            async def playback_reader() -> None:
                while not stop_tasks.is_set():
                    tick_start = time.monotonic()
                    outdata = np.zeros(FRAME_SAMPLES_24K, dtype=np.float32)
                    copied = playback.read_into(outdata)
                    if copied:
                        now_ns = time.monotonic_ns()
                        state["playback_samples_consumed"] += copied
                        cumulative = int(state["playback_samples_consumed"])
                        consumed_segments = shadow_consume(copied, now_ns)
                        self.event_logger.log(
                            "client_pcm_consumed",
                            speaker="assistant",
                            details={
                                "copied_samples": int(copied),
                                "cumulative_consumed_samples": cumulative,
                                "sample_rate": SAMPLE_RATE_OUTPUT,
                                "callback_samples": FRAME_SAMPLES_24K,
                                "consumption_semantics": "logical_client_device_buffer",
                                "consumed_segments": consumed_segments,
                            },
                        )
                        if state["first_pcm_consumed_ns"] is None:
                            state["first_pcm_consumed_ns"] = now_ns
                            first_pcm_consumed.set()
                        for ack in playback.drain_client_consumed_pcm_acks():
                            await ack_queue.put(ack)
                    elapsed = time.monotonic() - tick_start
                    await asyncio.sleep(max(0.0, self.config.frame_pacing_s - elapsed))

            async def ack_sender() -> None:
                while not stop_tasks.is_set() or not ack_queue.empty():
                    try:
                        ack = await asyncio.wait_for(ack_queue.get(), timeout=0.05)
                    except asyncio.TimeoutError:
                        continue
                    async with send_lock:
                        async with event_lock:
                            if not playback.accept_client_consumed_pcm_ack(ack):
                                self.event_logger.log(
                                    "client_consumed_pcm_ack_discarded",
                                    speaker="assistant",
                                    details=ack,
                                )
                                continue
                            await ws.send(json.dumps(ack))
                            sent_ns = time.monotonic_ns()
                        self.event_logger.log(
                            "raw_ws_send",
                            speaker="assistant",
                            details=ack,
                        )
                    state["consumed_acknowledgements"].append({**ack, "sent_ns": sent_ns})
                    if state["first_ack_sent_ns"] is None:
                        state["first_ack_sent_ns"] = sent_ns
                    self.event_logger.log(
                        "client_consumed_pcm_sent",
                        speaker="assistant",
                        details={**ack, "sent_ns": sent_ns},
                    )

            async def playback_silence_ticks() -> None:
                silence = np.zeros(FRAME_SAMPLES_16K, dtype=np.float32)
                while not stop_tasks.is_set():
                    await asyncio.sleep(self.config.frame_pacing_s)
                    if (
                        (state["pending_committed_steps"] or state["active_responses"]
                         or playback.pending_samples > 0)
                        and not state["input_active"]
                    ):
                        pending_turns = state["pending_committed_steps"]
                        if pending_turns:
                            step_id = max(pending_turns, key=pending_turns.get)
                            await send_input_frame(
                                silence,
                                source="committed_turn_silence",
                                step_id=step_id,
                                idle_only=True,
                            )
                        else:
                            await send_input_frame(
                                silence,
                                source="assistant_playback_silence",
                                idle_only=True,
                            )

            producer_task = asyncio.create_task(input_producer())
            playback_task = asyncio.create_task(playback_reader())
            ack_task = asyncio.create_task(ack_sender())
            silence_task = asyncio.create_task(playback_silence_ticks())
            last_event_ns = time.monotonic_ns()
            max_duration_ns = int(duration_limit_s * 1e9)
            quiet_timeout_ns = int(max(0.01, self.config.silence_timeout_s) * 1e9)

            try:
                while True:
                    now_ns = time.monotonic_ns()
                    elapsed_ns = now_ns - run_started_ns
                    if elapsed_ns >= max_duration_ns:
                        state["stop_reason"] = "max_duration"
                        break
                    if state["producer_errors"]:
                        state["stop_reason"] = "input_error"
                        break
                    completion_ready = (
                        producer_finished.is_set()
                        and state["last_commit_ns"] is not None
                        and state["last_terminal_ns"] is not None
                        and state["last_terminal_ns"] >= state["last_commit_ns"]
                        and not state["active_responses"]
                        and playback.pending_samples == 0
                        and ack_queue.empty()
                        and not state["pending_committed_steps"]
                    )
                    if completion_ready:
                        state["stop_reason"] = "response_terminal"
                        break
                    if (
                        producer_finished.is_set()
                        and not state["active_responses"]
                        and playback.pending_samples == 0
                        and not state["pending_committed_steps"]
                        and ack_queue.empty()
                        and now_ns - last_event_ns >= quiet_timeout_ns
                    ):
                        state["stop_reason"] = "quiet_timeout"
                        break
                    remaining_s = max(0.001, (max_duration_ns - elapsed_ns) / 1e9)
                    try:
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=min(0.05, remaining_s))
                    except asyncio.TimeoutError:
                        continue
                    except EOFError:
                        state["stop_reason"] = "websocket_closed"
                        break

                    received_ns = time.monotonic_ns()
                    last_event_ns = received_ns
                    state["last_event_ns"] = received_ns
                    msg = json.loads(raw_msg)
                    mtype = str(msg.get("type", "unknown"))
                    self.event_logger.log(mtype, speaker="assistant", details=msg)
                    response_id = msg.get("response_id")

                    if mtype == "response.created":
                        response_id = str(response_id or "")
                        if response_id:
                            async with event_lock:
                                playback.begin_response(response_id)
                                shadow_begin_response(response_id)
                            state["active_responses"].add(response_id)
                            pending_commits = [
                                (step_id, int(commit_ns))
                                for step_id, commit_ns in state["pending_committed_steps"].items()
                                if int(commit_ns) <= received_ns
                            ]
                            if pending_commits:
                                pending_step_id, _pending_commit_ns = max(
                                    pending_commits, key=lambda item: item[1]
                                )
                                state["response_pending_steps"][response_id] = pending_step_id
                            response = state["responses"].setdefault(response_id, {
                                "response_id": response_id,
                                "created_ns": received_ns,
                            "created_after_commit_ns": next(
                                (commit_ns for commit_ns in reversed(state["commit_timestamps_ns"])
                                 if commit_ns <= received_ns),
                                None,
                            ),
                                "text": "",
                                "audio_packets": [],
                            })
                            response["created_ns"] = received_ns
                            state["response_generations"].setdefault(response_id, set())
                        response["created_after_commit_ns"] = next(
                            (commit_ns for commit_ns in reversed(state["commit_timestamps_ns"])
                             if commit_ns <= received_ns),
                            None,
                        )
                        if state["first_response_created_ns"] is None:
                            state["first_response_created_ns"] = received_ns
                    elif mtype == "speaker.segment.opened":
                        generation_id = msg.get("generation_id")
                        if isinstance(generation_id, int) and not isinstance(generation_id, bool):
                            async with event_lock:
                                playback.advance_generation(generation_id)
                                shadow_advance_generation(generation_id)
                                if response_id:
                                    state["response_generations"].setdefault(str(response_id), set()).add(
                                        generation_id
                                    )
                    elif mtype == "response.audio.delta":
                        if state["first_audio_delta_ns"] is None:
                            state["first_audio_delta_ns"] = received_ns
                        packet = decode_playback_packet(msg)
                        pending_step_id = state["response_pending_steps"].get(packet.response_id)
                        if pending_step_id is None:
                            pending_commits = [
                                (step_id, int(commit_ns))
                                for step_id, commit_ns in state["pending_committed_steps"].items()
                                if int(commit_ns) <= received_ns
                            ]
                            if pending_commits:
                                pending_step_id, _pending_commit_ns = max(
                                    pending_commits, key=lambda item: item[1]
                                )
                                state["response_pending_steps"][packet.response_id] = pending_step_id
                        if pending_step_id in state["pending_committed_steps"]:
                            pending_commit_ns = state["pending_committed_steps"].pop(pending_step_id)
                            self.event_logger.log(
                                "committed_turn_first_audio_available",
                                speaker="assistant",
                                details={
                                    "step_id": pending_step_id,
                                    "commit_ns": pending_commit_ns,
                                    "response_id": packet.response_id,
                                    "first_audio_delta_received_ns": received_ns,
                                },
                            )
                        async with event_lock:
                            accepted = playback.enqueue_packet(
                                response_id=packet.response_id,
                                generation_id=packet.generation_id,
                                speaker_segment_id=packet.speaker_segment_id,
                                packet_id=packet.packet_id,
                                first_frame_id=packet.first_frame_id,
                                last_frame_id=packet.last_frame_id,
                                start_sample=packet.start_sample,
                                end_sample=packet.end_sample,
                                pcm=packet.pcm,
                                sample_rate=packet.sample_rate,
                            )
                            if accepted:
                                state["response_generations"].setdefault(packet.response_id, set()).add(
                                    packet.generation_id
                                )
                                shadow_advance_generation(packet.generation_id)
                                shadow_chunks.append({
                                    "response_id": packet.response_id,
                                    "generation_id": packet.generation_id,
                                    "speaker_segment_id": packet.speaker_segment_id,
                                    "start_sample": packet.start_sample,
                                    "end_sample": packet.end_sample,
                                    "sample_rate": packet.sample_rate,
                                    "offset": 0,
                                })
                        if accepted:
                            self.recorder.record_output("A", packet.pcm)
                            response = state["responses"].setdefault(packet.response_id, {
                                "response_id": packet.response_id,
                                "text": "",
                                "audio_packets": [],
                            })
                            response.setdefault("first_audio_delta_ns", received_ns)
                            response["audio_packets"].append({
                                "packet_id": packet.packet_id,
                                "generation_id": packet.generation_id,
                                "speaker_segment_id": packet.speaker_segment_id,
                                "start_sample": packet.start_sample,
                                "end_sample": packet.end_sample,
                                "sample_rate": packet.sample_rate,
                            })
                    elif mtype == "response.audio.segment_done":
                        async with event_lock:
                            playback.complete_segment(
                                response_id=str(msg.get("response_id", "")),
                                generation_id=int(msg["generation_id"]),
                                speaker_segment_id=str(msg.get("speaker_segment_id", "")),
                                start_sample=int(msg["start_sample"]),
                                end_sample=int(msg["end_sample"]),
                                sample_rate=int(msg["sample_rate"]),
                            )
                            for ack in playback.drain_client_consumed_pcm_acks():
                                await ack_queue.put(ack)
                    elif mtype in {
                        "response.output_text.delta",
                        "response.audio_transcript.delta",
                        "response.output_audio_transcript.delta",
                    }:
                        if response_id:
                            response = state["responses"].setdefault(str(response_id), {
                                "response_id": str(response_id),
                                "text": "",
                                "audio_packets": [],
                            })
                            response["text"] += str(msg.get("delta", ""))
                    elif mtype == "response.done":
                        if response_id:
                            async with event_lock:
                                playback.complete_response(str(response_id))
                            state["active_responses"].discard(str(response_id))
                            state["response_generations"].pop(str(response_id), None)
                            shadow_completed_responses.add(str(response_id))
                            if shadow_active_response == str(response_id):
                                shadow_active_response = None
                        state["last_terminal_ns"] = received_ns
                    elif mtype in {"response.cancelled", "audio.cancelled"}:
                        generation_id = msg.get("generation_id")
                        async with event_lock:
                            playback.cancel_response(
                                None if response_id is None else str(response_id),
                                int(generation_id) if generation_id is not None else None,
                            )
                            shadow_cancel_response(
                                None if response_id is None else str(response_id),
                                int(generation_id) if generation_id is not None else None,
                            )
                        if response_id:
                            response_key = str(response_id)
                            known_generations = state["response_generations"].get(response_key, set())
                            if generation_id is None:
                                known_generations.clear()
                            else:
                                known_generations.discard(int(generation_id))
                            if not known_generations:
                                state["response_generations"].pop(response_key, None)
                                state["active_responses"].discard(response_key)
                                self.timeline_a.cancel_response(response_key)
                        state["last_terminal_ns"] = received_ns
                        self.event_logger.log(
                            "response_stop_observed",
                            speaker="assistant",
                            details={
                                "response_id": response_id,
                                "generation_id": generation_id,
                                "reason": msg.get("reason", "cancelled"),
                                "timestamp_ns": received_ns,
                            },
                        )
                    elif mtype == "input_audio_buffer.consumed":
                        state["ingress_acks_observed"] = True
                        state["frames_consumed"] = state.get("frames_consumed", 0) + 1
                        frame_consumed_event.set()
                        self.event_logger.log(
                            "input_audio_frame_consumed",
                            speaker="user",
                            turn_index=int(msg.get("frame_id", -1)),
                            details={
                                "frame_id": msg.get("frame_id"),
                                "listener_output_frame_id": msg.get("listener_output_frame_id"),
                                "timestamp_ns": received_ns,
                            },
                        )
                    elif mtype == "input_audio_buffer.conditioning_available":
                        self.event_logger.log(
                            "input_audio_conditioning_available",
                            speaker="user",
                            turn_index=int(msg.get("frame_id", -1)),
                            details=msg,
                        )
                    elif mtype == "error":
                        error = msg.get("error", {})
                        state["server_errors"].append(
                            dict(error) if isinstance(error, dict) else {"message": str(error)}
                        )

                    for container in (msg, msg.get("policy"), msg.get("policy_diagnostics"), msg.get("details")):
                        if not isinstance(container, dict):
                            continue
                        policy_fields = {
                            key: container[key]
                            for key in (
                                "raw_action", "legal_action", "selected_action", "applied_action",
                                "policy_raw_action", "policy_legal_action",
                                "policy_selected_action", "policy_applied_action",
                            )
                            if key in container
                        }
                        if policy_fields:
                            state["policy_diagnostics"].append({
                                "timestamp_ns": received_ns,
                                "response_id": response_id,
                                **policy_fields,
                            })
                            self.event_logger.log(
                                "policy_diagnostics",
                                speaker="assistant",
                                details={"response_id": response_id, **policy_fields},
                            )
                            break
            finally:
                state["stop_ns"] = time.monotonic_ns()
                if state["stop_reason"] == "running":
                    state["stop_reason"] = "stopped"
                self.event_logger.log(
                    "streaming_scenario_stopped",
                    details={
                        "scenario_id": scenario.scenario_id,
                        "stop_reason": state["stop_reason"],
                        "timestamp_ns": state["stop_ns"],
                        "elapsed_ms": (state["stop_ns"] - run_started_ns) / 1e6,
                    },
                )
                stop_tasks.set()
                for task in (producer_task, playback_task, silence_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(producer_task, playback_task, silence_task, return_exceptions=True)
                await ack_task
                if transcription_tasks:
                    await asyncio.gather(*transcription_tasks, return_exceptions=True)

        wav_files = self.recorder.finalize()
        self.event_logger.close()
        first_pcm_ns = state["first_pcm_consumed_ns"]
        first_audio_ns = state["first_audio_delta_ns"]
        first_policy_speak_ns = next(
            (
                int(item["timestamp_ns"])
                for item in state["policy_diagnostics"]
                if str(item.get("applied_action", item.get("policy_applied_action", ""))).upper() == "SPEAK"
                and first_pcm_ns is not None
                and int(item["timestamp_ns"]) <= first_pcm_ns
            ),
            None,
        )
        correction_step = next(
            (
                (step_id, step_timing)
                for step_id, step_timing in state["step_schedule"].items()
                if step_timing.get("reference") == "first_client_pcm_consumed"
            ),
            None,
        )
        correction_timing: Optional[Dict[str, Any]] = None
        if correction_step is not None:
            correction_id, timing = correction_step
            commit_ns = timing.get("commit_ns")
            correction_responses = [
                response for response in state["responses"].values()
                if commit_ns is not None and response.get("created_after_commit_ns") == commit_ns
            ]
            correction_response_ids = {response["response_id"] for response in correction_responses}
            consumed_ack_ns = min(
                (
                    int(ack["client_monotonic_ns"])
                    for ack in state["consumed_acknowledgements"]
                    if ack.get("response_id") in correction_response_ids
                ),
                default=None,
            )
            correction_timing = {
                "step_id": correction_id,
                "started_ns": timing.get("actual_start_ns"),
                "first_sample_sent_ns": timing.get("first_sample_sent_ns"),
                "commit_ns": commit_ns,
                "old_response_ids_at_first_correction_sample": timing.get(
                    "old_response_ids_at_first_correction_sample", []
                ),
                "last_old_response_pcm_consumed_ns": max(
                    (
                        int(state["last_consumed_ns_by_response"][response_id])
                        for response_id in timing.get("old_response_ids_at_first_correction_sample", [])
                        if response_id in state["last_consumed_ns_by_response"]
                    ),
                    default=None,
                ),
                "first_response_created_ns": min(
                    (int(response["created_ns"]) for response in correction_responses),
                    default=None,
                ),
                "first_audio_delta_ns": min(
                    (int(response["first_audio_delta_ns"]) for response in correction_responses
                     if response.get("first_audio_delta_ns") is not None),
                    default=None,
                ),
                "first_consumed_segment_ack_ns": consumed_ack_ns,
                "commit_to_first_consumed_segment_ack_ms": (
                    (consumed_ack_ns - commit_ns) / 1e6 if consumed_ack_ns is not None else None
                ),
            }
            first_sample_sent_ns = correction_timing["first_sample_sent_ns"]
            last_old_consumed_ns = correction_timing["last_old_response_pcm_consumed_ns"]
            correction_timing["stop_latency_first_correction_sample_to_last_old_consumed_pcm_ms"] = (
                (last_old_consumed_ns - first_sample_sent_ns) / 1e6
                if first_sample_sent_ns is not None
                and last_old_consumed_ns is not None
                and last_old_consumed_ns >= first_sample_sent_ns
                else None
            )
        first_committed_response = min(
            (
                response for response in state["responses"].values()
                if response.get("created_after_commit_ns") is not None
            ),
            key=lambda response: int(response["created_ns"]),
            default=None,
        )
        result = {
            "status": (
                "COMPLETED" if state["stop_reason"] == "response_terminal"
                else "INCOMPLETE" if state["stop_reason"] == "quiet_timeout"
                else "FAILED" if (
                    state["stop_reason"] == "max_duration"
                    and state["pending_committed_steps"]
                )
                else "STOPPED"
            ),
            "mode": "single_session_streaming",
            "scenario_id": scenario.scenario_id,
            "stop_reason": state["stop_reason"],
            "input_steps": state["step_schedule"],
            "input_transcripts": state["transcripts"],
            "responses": list(state["responses"].values()),
            "consumed_acknowledgements": state["consumed_acknowledgements"],
            "policy_diagnostics": state["policy_diagnostics"],
            "server_errors": state["server_errors"],
            "committed_turns_awaiting_first_audio": sorted(state["pending_committed_steps"]),
            "playback_shadow_tracker_errors": state["shadow_tracker_errors"],
            "last_consumed_pcm_ns_by_response": state["last_consumed_ns_by_response"],
            "timing": {
                "scenario_started_ns": run_started_ns,
                "first_response_created_ns": state["first_response_created_ns"],
                "first_response_created_after_commit_ms": (
                    (int(first_committed_response["created_ns"])
                     - int(first_committed_response["created_after_commit_ns"])) / 1e6
                    if first_committed_response is not None
                    else None
                ),
                "first_audio_delta_ns": first_audio_ns,
                "first_audio_delta_after_response_created_ms": (
                    (first_audio_ns - state["first_response_created_ns"]) / 1e6
                    if first_audio_ns is not None and state["first_response_created_ns"] is not None else None
                ),
                "first_client_pcm_consumed_ns": first_pcm_ns,
                "policy_speak_to_first_client_pcm_consumed_ms": (
                    (first_pcm_ns - first_policy_speak_ns) / 1e6
                    if first_pcm_ns is not None and first_policy_speak_ns is not None else None
                ),
                "correction_timing": correction_timing,
                "first_client_consumed_pcm_ack_sent_ns": state["first_ack_sent_ns"],
                "last_response_terminal_ns": state["last_terminal_ns"],
                "stop_ns": state["stop_ns"],
                "client_consumption_semantics": "logical_client_device_buffer",
            },
            "ingress_frames_sent": int(state["input_frame_id"]),
            "logical_pcm_samples_consumed": int(state["playback_samples_consumed"]),
            "wav_files": {key: str(path) for key, path in wav_files.items()},
            "events_jsonl": str(self.output_dir / "dialogue_events.jsonl"),
        }
        report_path = self.output_dir / "streaming_dialogue_report.json"
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["report_json"] = str(report_path)
        return result

    async def run_mode_a(
        self,
        scenario: Any,
        audio_inputs: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Mode (a): User audio playback (questions, clarifications, interruptions) vs Assistant."""
        from dialogue_eval.scenarios import generate_speech_for_text

        self.event_logger.log("mode_a_started", details={"scenario_id": scenario.scenario_id})
        audio_inputs = audio_inputs or {}

        ws_ctx = await self._connect_ws(self.config.ws_url)
        async with ws_ctx as ws:
            await self._setup_session(
                ws,
                speaker_name="A",
                voice=self.config.voice_b,
                instructions=scenario.system_prompt_b,
                turn_policy=self.config.turn_policy,
                auto_response=False,
            )

            current_turn = 0
            server_errors: List[Dict[str, Any]] = []
            input_reference_by_turn: Dict[int, str] = {}
            input_transcript_by_turn: Dict[int, str] = {}
            input_asr_available_by_turn: Dict[int, bool] = {}
            for step in scenario.steps:
                if step.speaker == "user":
                    current_turn += 1
                    # Prefer an explicitly supplied array, then a cached Qwen3-TTS WAV.
                    pcm = audio_inputs.get(step.step_id)
                    if pcm is not None:
                        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
                        audio_source = "provided_pcm"
                        self.event_logger.log(
                            "scenario_audio_provided",
                            speaker="user",
                            turn_index=current_turn,
                            details={"step_id": step.step_id, "samples": int(pcm.size)},
                        )
                    else:
                        cached_path = self.scenario_audio_dir / f"{scenario.scenario_id}__{step.step_id}.wav"
                        if cached_path.is_file():
                            pcm = read_scenario_audio_wav(cached_path)
                            audio_source = "qwen3tts_cache"
                            self.event_logger.log(
                                "scenario_audio_loaded",
                                speaker="user",
                                turn_index=current_turn,
                                details={"step_id": step.step_id, "path": str(cached_path), "samples": int(pcm.size)},
                            )
                        else:
                            pcm = generate_speech_for_text(step.expected_text)
                            audio_source = "synthetic_fallback"
                            self.event_logger.log(
                                "scenario_audio_fallback_synthetic",
                                speaker="user",
                                turn_index=current_turn,
                                details={"step_id": step.step_id, "samples": int(pcm.size)},
                            )
                    if pcm.size == 0 or not np.isfinite(pcm).all():
                        raise ValueError(f"Invalid input PCM for scenario step {step.step_id}")
                    input_reference_by_turn[current_turn] = step.expected_text

                    frames, remainder = pcm_to_80ms_frames(pcm)
                    if remainder.size:
                        padded_frame = np.zeros(FRAME_SAMPLES_16K, dtype=np.float32)
                        padded_frame[:remainder.size] = remainder
                        frames.append(padded_frame)
                    pcm_sent = np.concatenate(frames) if frames else np.empty(0, dtype=np.float32)
                    self.recorder.record_input("A", pcm_sent)

                    self.event_logger.log(
                        "user_speech_start",
                        speaker="user",
                        turn_index=current_turn,
                        details={"step_id": step.step_id, "frames": len(frames)},
                    )
                    for frame in frames:
                        pcm16_bytes = float_to_pcm16_bytes(frame)
                        b64_audio = base64.b64encode(pcm16_bytes).decode("ascii")
                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": b64_audio,
                        }))
                        await asyncio.sleep(self.config.frame_pacing_s)
                    self.event_logger.log("user_speech_end", speaker="user", turn_index=current_turn)
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    transcript, asr_available, asr_error = await self._transcribe_input_audio(pcm_sent)
                    input_transcript_by_turn[current_turn] = transcript
                    input_asr_available_by_turn[current_turn] = asr_available
                    self.event_logger.log(
                        "input_audio_transcribed",
                        speaker="user",
                        turn_index=current_turn,
                        details={
                            "step_id": step.step_id,
                            "transcript": transcript,
                            "asr_available": asr_available,
                            "audio_source": audio_source,
                            **({"asr_error": asr_error} if asr_error else {}),
                        },
                    )

                elif step.speaker == "assistant":
                    # Collect assistant response while streaming ambient silence at 80 ms to keep listener alive
                    accumulated_text = ""
                    received_audio_chunks: List[np.ndarray] = []
                    ttfa_ms: Optional[float] = None
                    t_start = time.monotonic()

                    stop_silence = asyncio.Event()
                    silence_frame = np.zeros(FRAME_SAMPLES_16K, dtype=np.float32)
                    silence_b64 = base64.b64encode(float_to_pcm16_bytes(silence_frame)).decode("ascii")
                    silence_msg = json.dumps({"type": "input_audio_buffer.append", "audio": silence_b64})

                    async def _feed_ambient_silence() -> None:
                        while not stop_silence.is_set():
                            try:
                                await asyncio.sleep(self.config.frame_pacing_s)
                                if stop_silence.is_set():
                                    break
                                await ws.send(silence_msg)
                            except Exception:
                                break

                    silence_task = asyncio.create_task(_feed_ambient_silence())
                    try:
                        while True:
                            try:
                                raw_msg = await asyncio.wait_for(ws.recv(), timeout=self.config.silence_timeout_s)
                            except (asyncio.TimeoutError, EOFError):
                                break
                            msg = json.loads(raw_msg)
                            mtype = msg.get("type")
                            self.event_logger.log(mtype, speaker="assistant", turn_index=current_turn, details=msg)

                            if mtype == "response.output_text.delta":
                                accumulated_text += msg.get("delta", "")
                            elif mtype == "response.audio.delta":
                                if ttfa_ms is None:
                                    ttfa_ms = (time.monotonic() - t_start) * 1000.0
                                packet = decode_response_audio(msg)
                                self.timeline_a.add_packet(
                                    response_id=packet.response_id,
                                    generation_id=packet.generation_id,
                                    packet_id=packet.packet_id,
                                    first_frame_id=packet.first_frame_id,
                                    last_frame_id=packet.last_frame_id,
                                    start_sample=packet.start_sample,
                                    end_sample=packet.end_sample,
                                    pcm=packet.pcm,
                                )
                                received_audio_chunks.append(packet.pcm)
                                self.recorder.record_output("A", packet.pcm)
                            elif mtype == "response.done":
                                break
                            elif mtype == "response.cancelled":
                                self.timeline_a.cancel_response(msg.get("response_id", ""))
                                break
                            elif mtype == "error":
                                error = msg.get("error", {})
                                normalized_error = dict(error) if isinstance(error, dict) else {"message": str(error)}
                                server_errors.append(normalized_error)
                                self.event_logger.log(
                                    "assistant_server_error",
                                    speaker="assistant",
                                    turn_index=current_turn,
                                    details=normalized_error,
                                )
                                break
                    finally:
                        stop_silence.set()
                        silence_task.cancel()
                        try:
                            await silence_task
                        except (asyncio.CancelledError, Exception):
                            pass

                    coherent_pcm = np.concatenate(received_audio_chunks) if received_audio_chunks else np.empty(0, dtype=np.float32)

                    # Evaluate turn
                    self.evaluator.evaluate_turn(
                        turn_index=current_turn,
                        speaker="assistant",
                        scenario=scenario,
                        step=step,
                        recognized_input=input_transcript_by_turn.get(current_turn, ""),
                        generated_text=accumulated_text,
                        recognized_output=accumulated_text,
                        audio_pcm=coherent_pcm,
                        sample_rate=SAMPLE_RATE_OUTPUT,
                        timing_metrics={"ttfa_ms": ttfa_ms or 50.0},
                        input_reference_text=input_reference_by_turn.get(current_turn, ""),
                        input_asr_available=input_asr_available_by_turn.get(current_turn, False),
                    )

                    if current_turn >= self.config.max_turns:
                        break

            saved_wavs = self.recorder.finalize()
            self.event_logger.close()
            summary = self.evaluator.build_summary_report()
            md_report = self.evaluator.generate_markdown_report()
            (self.output_dir / "dialogue_report.md").write_text(md_report, encoding="utf-8")
            (self.output_dir / "summary_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

            return {
                "status": "FAILED" if server_errors else "COMPLETED",
                "server_errors": server_errors,
                "report": summary,
                "markdown_report": md_report,
                "wav_files": saved_wavs,
            }

    async def run_mode_b(
        self,
        scenario: Any,
    ) -> Dict[str, Any]:
        """Mode (b): Autonomous spoken dialogue between Interlocutors A and B over WebSockets."""
        from dialogue_eval.scenarios import generate_speech_for_text

        self.event_logger.log("mode_b_started", details={"scenario_id": scenario.scenario_id})

        ws_a_ctx = await self._connect_ws(self.config.ws_url)
        ws_b_ctx = await self._connect_ws(self.config.ws_url)

        async with ws_a_ctx as ws_a, ws_b_ctx as ws_b:
            await self._setup_session(
                ws_a,
                speaker_name="A",
                voice=self.config.voice_a,
                instructions=scenario.system_prompt_a,
                turn_policy=self.config.turn_policy,
                auto_response=False,
                input_audio_buffer_acknowledgements=True,
                input_audio_buffer_conditioning_acknowledgements=True,
            )
            await self._setup_session(
                ws_b,
                speaker_name="B",
                voice=self.config.voice_b,
                instructions=scenario.system_prompt_b,
                turn_policy=self.config.turn_policy,
                auto_response=False,
                input_audio_buffer_acknowledgements=True,
                input_audio_buffer_conditioning_acknowledgements=True,
            )

            self.event_logger.log("sessions_initialized", details={
                "session_a_voice": self.config.voice_a,
                "session_b_voice": self.config.voice_b,
            })

            stop_event = asyncio.Event()
            turns_completed = 0
            t_start = time.monotonic()
            accumulated_text_a = ""
            accumulated_text_b = ""
            responses_by_id: Dict[Tuple[str, str], Dict[str, Any]] = {}
            latest_completed_response: Dict[str, Dict[str, Any]] = {}
            input_onsets_ns: List[Dict[str, Any]] = []
            turn_transfers: List[Dict[str, Any]] = []
            ingress_failures: List[str] = []
            ack_queues = {"A": asyncio.Queue(maxsize=1), "B": asyncio.Queue(maxsize=1)}
            conditioning_ack_queues = {"A": asyncio.Queue(maxsize=1), "B": asyncio.Queue(maxsize=1)}
            next_server_frame_id = {"A": 0, "B": 0}
            inflight_ack_expected: Dict[str, int | None] = {"A": None, "B": None}
            ack_received_for_inflight = {"A": False, "B": False}
            conditioning_ack_received_for_inflight = {"A": False, "B": False}
            inflight_ack_tasks: Dict[str, asyncio.Task | None] = {"A": None, "B": None}
            inflight_ack_results: Dict[str, bool | None] = {"A": None, "B": None}
            abandoned_ack_frames: Dict[str, Dict[int, set[str]]] = {"A": {}, "B": {}}
            abandoned_ack_drained = {"A": asyncio.Event(), "B": asyncio.Event()}
            abandoned_ack_drained["A"].set()
            abandoned_ack_drained["B"].set()
            ack_settled_events = {"A": asyncio.Event(), "B": asyncio.Event()}
            ack_settled_events["A"].set()
            ack_settled_events["B"].set()
            commit_waiters: Dict[str, asyncio.Future] = {}
            bridge_input_chunks: Dict[str, List[np.ndarray]] = {"A": [], "B": []}
            response_done_events: Dict[Tuple[str, str], asyncio.Event] = {}
            response_cancelled_events: Dict[Tuple[str, str], asyncio.Event] = {}
            latest_response_ids: Dict[str, str | None] = {"A": None, "B": None}
            response_created_events = {"A": asyncio.Event(), "B": asyncio.Event()}
            sockets = {"A": ws_a, "B": ws_b}

            def _fail_transport(reason: str, *, speaker: str | None = None, details: Dict[str, Any] | None = None) -> None:
                ingress_failures.append(reason)
                self.event_logger.log(
                    "mode_b_transport_failure",
                    speaker=speaker,
                    turn_index=turns_completed + 1,
                    details={"reason": reason, **(details or {})},
                )
                stop_event.set()

            def _clear_inflight_ack(speaker: str) -> None:
                inflight_ack_expected[speaker] = None
                ack_received_for_inflight[speaker] = False
                conditioning_ack_received_for_inflight[speaker] = False
                for pending_queue in (ack_queues[speaker], conditioning_ack_queues[speaker]):
                    while not pending_queue.empty():
                        try:
                            pending_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                ack_settled_events[speaker].set()

            def _ack_timeout_s(source: str) -> float:
                timeout_s = max(0.0, float(self.config.ingress_ack_timeout_s))
                if source == "conditioning_silence":
                    remaining_s = self.config.max_duration_s - (time.monotonic() - t_start)
                    timeout_s = min(timeout_s, max(0.0, remaining_s))
                return timeout_s

            async def _settle_sent_frame_ack(
                speaker: str,
                expected_frame_id: int,
                sent_at_ns: int,
                source: str,
            ) -> bool:
                nonlocal stop_reason
                ack_queue = ack_queues[speaker]
                conditioning_ack_queue = conditioning_ack_queues[speaker]
                succeeded = False
                try:
                    try:
                        ack = await asyncio.wait_for(
                            ack_queue.get(), timeout=_ack_timeout_s(source)
                        )
                    except asyncio.TimeoutError:
                        raise TimeoutError("listener ACK timeout")
                    try:
                        received_frame_id = int(ack.get("frame_id", -1))
                    except (OverflowError, TypeError, ValueError):
                        received_frame_id = -1
                    if received_frame_id != expected_frame_id:
                        _clear_inflight_ack(speaker)
                        _fail_transport(
                            f"{speaker} out-of-order listener ACK: expected {expected_frame_id}, got {received_frame_id}",
                            speaker=speaker,
                            details={"expected_frame_id": expected_frame_id, "received_frame_id": received_frame_id},
                        )
                        return False
                    try:
                        conditioning_ack = await asyncio.wait_for(
                            conditioning_ack_queue.get(), timeout=_ack_timeout_s(source)
                        )
                    except asyncio.TimeoutError:
                        raise TimeoutError("conditioning frame notification timeout")
                    try:
                        conditioning_frame_id = int(conditioning_ack.get("frame_id", -1))
                        published_frame_id = int(conditioning_ack.get("conditioning_frame_id", -1))
                        listener_output_frame_id = int(conditioning_ack.get("listener_output_frame_id", -1))
                    except (OverflowError, TypeError, ValueError):
                        conditioning_frame_id = published_frame_id = listener_output_frame_id = -1
                    if (
                        conditioning_frame_id != expected_frame_id
                        or published_frame_id != expected_frame_id
                        or listener_output_frame_id != expected_frame_id
                        or conditioning_ack.get("conditioning_frame_available") is not True
                    ):
                        _clear_inflight_ack(speaker)
                        _fail_transport(
                            f"{speaker} listener did not publish Qwen conditioning frame {expected_frame_id}",
                            speaker=speaker,
                            details=conditioning_ack,
                        )
                        return False
                    next_server_frame_id[speaker] += 1
                    inflight_ack_expected[speaker] = None
                    ack_received_for_inflight[speaker] = False
                    conditioning_ack_received_for_inflight[speaker] = False
                    acked_at_ns = time.monotonic_ns()
                    self.event_logger.log(
                        "input_audio_frame_consumed",
                        speaker=speaker,
                        turn_index=turns_completed + 1,
                        details={
                            "frame_id": received_frame_id,
                            "listener_output_frame_id": listener_output_frame_id,
                            "conditioning_frame_available": conditioning_ack.get("conditioning_frame_available"),
                            "ack_latency_ms": round((acked_at_ns - sent_at_ns) / 1_000_000.0, 3),
                            "source": source,
                        },
                    )
                    delay_s = self.config.frame_pacing_s - (acked_at_ns - sent_at_ns) / 1_000_000_000.0
                    if delay_s > 0.0:
                        await asyncio.sleep(delay_s)
                    succeeded = True
                    return True
                except TimeoutError as exc:
                    if source == "conditioning_silence" and (
                        stop_reason == "max_duration"
                        or self.config.max_duration_s - (time.monotonic() - t_start) <= 0.0
                    ):
                        stop_reason = "max_duration"
                        stop_event.set()
                        seen = set()
                        if ack_received_for_inflight[speaker]:
                            seen.add("consumed")
                        if conditioning_ack_received_for_inflight[speaker]:
                            seen.add("conditioning_available")
                        if seen != {"consumed", "conditioning_available"}:
                            abandoned_ack_frames[speaker][expected_frame_id] = seen
                            abandoned_ack_drained[speaker].clear()
                        _clear_inflight_ack(speaker)
                        return False
                    _clear_inflight_ack(speaker)
                    _fail_transport(
                        f"{speaker} {exc} for server frame {expected_frame_id}",
                        speaker=speaker,
                        details={"expected_frame_id": expected_frame_id},
                    )
                    return False
                finally:
                    inflight_ack_results[speaker] = succeeded
                    if inflight_ack_tasks[speaker] is asyncio.current_task():
                        inflight_ack_tasks[speaker] = None
                    ack_settled_events[speaker].set()

            async def _send_input_frame(
                speaker: str,
                frame: np.ndarray,
                *,
                source: str,
                interrupt_events: Tuple[asyncio.Event, ...] = (),
            ) -> bool | None:
                pcm = np.asarray(frame, dtype=np.float32).reshape(-1)
                if pcm.size != FRAME_SAMPLES_16K:
                    _fail_transport(
                        f"{speaker} frame size {pcm.size} != {FRAME_SAMPLES_16K}",
                        speaker=speaker,
                    )
                    return False
                ack_queue = ack_queues[speaker]
                conditioning_ack_queue = conditioning_ack_queues[speaker]
                if inflight_ack_expected[speaker] is not None:
                    settle_task = inflight_ack_tasks[speaker]
                    if settle_task is None:
                        _fail_transport(
                            f"{speaker} has an unowned in-flight ingress acknowledgement",
                            speaker=speaker,
                        )
                        return False
                    wait_timeout_s = max(0.1, 2.0 * float(self.config.ingress_ack_timeout_s) + 0.1)
                    await _wait_for_any_event(
                        (ack_settled_events[speaker], *interrupt_events), wait_timeout_s
                    )
                    if not ack_settled_events[speaker].is_set():
                        if any(event.is_set() for event in interrupt_events):
                            return None
                        _fail_transport(
                            f"{speaker} outstanding listener ACK did not settle before next frame",
                            speaker=speaker,
                        )
                        return False
                    if not await settle_task:
                        return False
                if (
                    not ack_queue.empty()
                    or not conditioning_ack_queue.empty()
                    or inflight_ack_expected[speaker] is not None
                ):
                    _fail_transport(f"{speaker} has an unexpected stale ingress acknowledgement", speaker=speaker)
                    return False
                expected_frame_id = next_server_frame_id[speaker]
                inflight_ack_expected[speaker] = expected_frame_id
                ack_received_for_inflight[speaker] = False
                conditioning_ack_received_for_inflight[speaker] = False
                inflight_ack_results[speaker] = None
                ack_settled_events[speaker].clear()
                sent_at_ns = time.monotonic_ns()
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(float_to_pcm16_bytes(pcm)).decode("ascii"),
                }
                try:
                    await sockets[speaker].send(json.dumps(message))
                except Exception as exc:
                    inflight_ack_results[speaker] = False
                    _clear_inflight_ack(speaker)
                    _fail_transport(f"{speaker} input send failed: {exc}", speaker=speaker)
                    return False

                if source in {"kickoff", "bridge"}:
                    self.recorder.record_input(speaker, pcm)
                    if source == "bridge":
                        bridge_input_chunks[speaker].append(pcm.copy())
                    self.event_logger.log(
                        "input_audio_frame_sent",
                        speaker=speaker,
                        turn_index=turns_completed + 1,
                        details={"source": source, "expected_server_frame_id": expected_frame_id, "frame_samples": int(pcm.size)},
                    )
                    input_onsets_ns.append({
                        "speaker": speaker,
                        "timestamp_ns": sent_at_ns,
                        "source": "bridged_audio_frame_sent" if source == "bridge" else "kickoff_audio_frame_sent",
                    })
                else:
                    self.event_logger.log(
                        "conditioning_silence_frame_sent",
                        speaker=speaker,
                        turn_index=turns_completed + 1,
                        details={"expected_server_frame_id": expected_frame_id, "frame_samples": int(pcm.size)},
                    )

                settle_task = asyncio.create_task(
                    _settle_sent_frame_ack(speaker, expected_frame_id, sent_at_ns, source)
                )
                inflight_ack_tasks[speaker] = settle_task
                if interrupt_events:
                    wait_timeout_s = max(0.1, 2.0 * float(self.config.ingress_ack_timeout_s) + 0.1)
                    await _wait_for_any_event(
                        (ack_settled_events[speaker], *interrupt_events), wait_timeout_s
                    )
                    if not ack_settled_events[speaker].is_set():
                        if any(event.is_set() for event in interrupt_events):
                            return None
                        await settle_task
                else:
                    await settle_task
                if not ack_settled_events[speaker].is_set():
                    _fail_transport(
                        f"{speaker} listener ACK wait ended without settling server frame {expected_frame_id}",
                        speaker=speaker,
                    )
                    return False
                return bool(inflight_ack_results[speaker])

            async def _commit_turn(speaker: str) -> tuple[Dict[str, Any] | None, int]:
                ws = sockets[speaker]
                for attempt in range(1, 4):
                    waiter = asyncio.get_running_loop().create_future()
                    commit_waiters[speaker] = waiter
                    try:
                        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                        self.event_logger.log(
                            "input_audio_buffer.commit_sent",
                            speaker=speaker,
                            turn_index=turns_completed + 1,
                            details={"attempt": attempt, "frames_acknowledged": next_server_frame_id[speaker]},
                        )
                        event = await asyncio.wait_for(waiter, timeout=self.config.silence_timeout_s)
                    except asyncio.TimeoutError:
                        _fail_transport(f"{speaker} response.created timeout after commit attempt {attempt}", speaker=speaker)
                        return None, attempt
                    except Exception as exc:
                        _fail_transport(f"{speaker} commit failed: {exc}", speaker=speaker)
                        return None, attempt
                    finally:
                        if commit_waiters.get(speaker) is waiter:
                            commit_waiters.pop(speaker, None)
                    if event.get("type") == "response.created":
                        return event, attempt
                    error = event.get("error", {})
                    if error.get("code") == "RESPONSE_ACTIVE":
                        self.event_logger.log(
                            "commit_retry_response_active",
                            speaker=speaker,
                            turn_index=turns_completed + 1,
                            details={"attempt": attempt, "response_id": event.get("response_id")},
                        )
                        await asyncio.sleep(max(0.01, self.config.frame_pacing_s))
                        continue
                    _fail_transport(f"{speaker} commit rejected: {error.get('code', 'unknown error')}", speaker=speaker, details=error)
                    return None, attempt
                _fail_transport(f"{speaker} commit retries exhausted (RESPONSE_ACTIVE)", speaker=speaker)
                return None, 3

            async def _wait_for_any_event(events: Tuple[asyncio.Event, ...], timeout_s: float) -> None:
                waiters = [asyncio.create_task(event.wait()) for event in events]
                try:
                    await asyncio.wait(
                        waiters,
                        timeout=max(0.0, float(timeout_s)),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    for waiter in waiters:
                        if not waiter.done():
                            waiter.cancel()
                    await asyncio.gather(*waiters, return_exceptions=True)

            async def _recv_loop(speaker: str, ws: Any, timeline: AudioTimelineReconstructor) -> None:
                nonlocal turns_completed, accumulated_text_a, accumulated_text_b
                while (
                    not stop_event.is_set()
                    or inflight_ack_expected[speaker] is not None
                    or bool(abandoned_ack_frames[speaker])
                ):
                    try:
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as exc:
                        if not stop_event.is_set():
                            _fail_transport(f"{speaker} WebSocket receive failed: {exc}", speaker=speaker)
                        return
                    try:
                        msg = json.loads(raw_msg)
                    except (TypeError, json.JSONDecodeError) as exc:
                        _fail_transport(f"{speaker} invalid server JSON: {exc}", speaker=speaker)
                        return
                    if not isinstance(msg, dict):
                        _fail_transport(f"{speaker} server event is not an object", speaker=speaker)
                        return
                    mtype = msg.get("type")
                    now_ns = time.monotonic_ns()
                    self.event_logger.log(mtype, speaker=speaker, turn_index=turns_completed + 1, details=msg)

                    if mtype == "input_audio_buffer.consumed":
                        try:
                            frame_id = int(msg.get("frame_id", -1))
                        except (OverflowError, TypeError, ValueError):
                            frame_id = -1
                        abandoned_seen = abandoned_ack_frames[speaker].get(frame_id)
                        if abandoned_seen is not None:
                            if "consumed" in abandoned_seen:
                                _fail_transport(
                                    f"{speaker} duplicate abandoned ingress ACK {frame_id}",
                                    speaker=speaker,
                                    details=msg,
                                )
                                return
                            abandoned_seen.add("consumed")
                            if abandoned_seen == {"consumed", "conditioning_available"}:
                                abandoned_ack_frames[speaker].pop(frame_id, None)
                                if not abandoned_ack_frames[speaker]:
                                    abandoned_ack_drained[speaker].set()
                            continue
                        expected = inflight_ack_expected[speaker]
                        if expected is None or ack_received_for_inflight[speaker] or frame_id != expected:
                            _fail_transport(
                                f"{speaker} unexpected, duplicate or out-of-order ingress ACK {frame_id}; expected {expected}",
                                speaker=speaker,
                                details=msg,
                            )
                            return
                        ack_received_for_inflight[speaker] = True
                        try:
                            ack_queues[speaker].put_nowait(msg)
                        except asyncio.QueueFull:
                            _fail_transport(f"{speaker} duplicate/unconsumed ingress ACK", speaker=speaker, details=msg)
                            return
                        continue
                    if mtype == "input_audio_buffer.conditioning_available":
                        try:
                            frame_id = int(msg.get("frame_id", -1))
                        except (OverflowError, TypeError, ValueError):
                            frame_id = -1
                        if not _conditioning_ack_matches_frame(msg, frame_id):
                            _fail_transport(
                                f"{speaker} listener published invalid conditioning frame {frame_id}",
                                speaker=speaker,
                                details=msg,
                            )
                            return
                        abandoned_seen = abandoned_ack_frames[speaker].get(frame_id)
                        if abandoned_seen is not None:
                            if "conditioning_available" in abandoned_seen:
                                _fail_transport(
                                    f"{speaker} duplicate abandoned conditioning notification {frame_id}",
                                    speaker=speaker,
                                    details=msg,
                                )
                                return
                            abandoned_seen.add("conditioning_available")
                            if abandoned_seen == {"consumed", "conditioning_available"}:
                                abandoned_ack_frames[speaker].pop(frame_id, None)
                                if not abandoned_ack_frames[speaker]:
                                    abandoned_ack_drained[speaker].set()
                            continue
                        expected = inflight_ack_expected[speaker]
                        if (
                            expected is None
                            or conditioning_ack_received_for_inflight[speaker]
                            or frame_id != expected
                        ):
                            _fail_transport(
                                f"{speaker} unexpected, duplicate or out-of-order conditioning notification "
                                f"{frame_id}; expected {expected}",
                                speaker=speaker,
                                details=msg,
                            )
                            return
                        conditioning_ack_received_for_inflight[speaker] = True
                        try:
                            conditioning_ack_queues[speaker].put_nowait(msg)
                        except asyncio.QueueFull:
                            _fail_transport(
                                f"{speaker} duplicate/unconsumed conditioning notification",
                                speaker=speaker,
                                details=msg,
                            )
                            return
                        continue
                    if mtype == "error":
                        waiter = commit_waiters.get(speaker)
                        if msg.get("error", {}).get("code") == "RESPONSE_ACTIVE" and waiter is not None and not waiter.done():
                            waiter.set_result(msg)
                            continue
                        _fail_transport(
                            f"{speaker} server error: {msg.get('error', {}).get('code', 'unknown error')}",
                            speaker=speaker,
                            details=msg.get("error", {}),
                        )
                        return

                    response_id = str(msg.get("response_id", ""))
                    response_key = (speaker, response_id)
                    if mtype == "response.created":
                        if response_key in responses_by_id:
                            _fail_transport(
                                f"{speaker} duplicate response.created for response ID {response_id}",
                                speaker=speaker,
                                details={"response_id": response_id},
                            )
                            return
                        responses_by_id[response_key] = {
                            "speaker": speaker,
                            "response_id": response_id,
                            "response_created_ns": now_ns,
                            "first_audio_ns": None,
                            "response_done_ns": None,
                            "cancelled_ns": None,
                            "generated_text": "",
                            "audio_samples": 0,
                        }
                        latest_response_ids[speaker] = response_id
                        response_done_events[response_key] = asyncio.Event()
                        response_cancelled_events[response_key] = asyncio.Event()
                        response_created_events[speaker].set()
                        waiter = commit_waiters.get(speaker)
                        if waiter is not None and not waiter.done():
                            waiter.set_result(msg)
                    elif mtype == "response.output_text.delta":
                        delta = msg.get("delta", "")
                        if response_key in responses_by_id:
                            responses_by_id[response_key]["generated_text"] += delta
                        if speaker == "A":
                            accumulated_text_a += delta
                        else:
                            accumulated_text_b += delta
                    elif mtype == "response.audio.delta":
                        try:
                            packet = decode_response_audio(msg)
                            accepted = timeline.add_packet(
                                response_id=packet.response_id,
                                generation_id=packet.generation_id,
                                packet_id=packet.packet_id,
                                first_frame_id=packet.first_frame_id,
                                last_frame_id=packet.last_frame_id,
                                start_sample=packet.start_sample,
                                end_sample=packet.end_sample,
                                pcm=packet.pcm,
                            )
                            if accepted and response_key in responses_by_id:
                                response = responses_by_id[response_key]
                                response["first_audio_ns"] = response["first_audio_ns"] or now_ns
                                response["last_audio_delta_ns"] = now_ns
                                response["audio_samples"] += int(packet.pcm.size)
                            if accepted:
                                self.recorder.record_output(speaker, packet.pcm)
                                self.bridge.feed_output(source_speaker=speaker, pcm_24k=packet.pcm)
                        except (ValueError, OverflowError, KeyError) as exc:
                            _fail_transport(f"{speaker} invalid/overflowed response audio: {exc}", speaker=speaker)
                            return
                    elif mtype == "response.cancelled":
                        if response_key in responses_by_id:
                            response_state = responses_by_id[response_key]
                            # The first terminal event wins. A late cancellation must not
                            # invalidate audio that has already completed; conversely,
                            # response.done after cancellation cannot revive that turn.
                            if response_state["response_done_ns"] is None and response_state["cancelled_ns"] is None:
                                response_state["cancelled_ns"] = now_ns
                                response_state["cancelled_reason"] = str(msg.get("reason", "unknown"))
                                response_cancelled_events[response_key].set()
                                timeline.cancel_response(response_id)
                                opposing = "B" if speaker == "A" else "A"
                                discarded = self.bridge.clear_channel(opposing)
                                self.event_logger.log("audio_bridge_cancelled_frames_discarded", speaker=speaker, details=discarded)
                    elif mtype == "response.done":
                        if response_key not in responses_by_id:
                            _fail_transport(f"{speaker} response.done without response.created: {response_id}", speaker=speaker)
                            return
                        response_state = responses_by_id[response_key]
                        if response_state["cancelled_ns"] is not None or response_state["response_done_ns"] is not None:
                            # Ignore a non-first terminal event for this response generation.
                            continue
                        try:
                            padded_frames = self.bridge.finalize_response(speaker)
                        except OverflowError as exc:
                            _fail_transport(str(exc), speaker=speaker)
                            return
                        response_state["response_done_ns"] = now_ns
                        response_state["final_partial_frames_padded"] = padded_frames
                        latest_completed_response[speaker] = response_state
                        response_done_events[response_key].set()
                        turns_completed += 1
                        if turns_completed >= self.config.max_turns:
                            stop_event.set()

            active_speaker = "A"
            transfer_frames_sent = 0
            stop_reason = ""

            async def _pacing_ingress_loop() -> None:
                nonlocal active_speaker, transfer_frames_sent, stop_reason
                while not stop_event.is_set():
                    if time.monotonic() - t_start >= self.config.max_duration_s:
                        stop_reason = "max_duration"
                        stop_event.set()
                        return
                    # Mode B bridges responses serially; only the active speaker receives
                    # exact-zero conditioning frames while its created response is live.
                    source_response_id = latest_response_ids[active_speaker]
                    source_response_key = (active_speaker, source_response_id) if source_response_id else None
                    source_done = response_done_events.get(source_response_key) if source_response_key else None
                    source_cancelled = response_cancelled_events.get(source_response_key) if source_response_key else None
                    if source_done is None or source_cancelled is None:
                        _fail_transport(
                            f"{active_speaker} has no tracked response generation to await",
                            speaker=active_speaker,
                            details={"response_id": source_response_id},
                        )
                        return
                    # Cancellation is terminal even if a later server event races with
                    # the pacing task; the receive loop enforces first-terminal-wins.
                    if source_cancelled.is_set():
                        if latest_response_ids[active_speaker] != source_response_id:
                            continue
                        response_created = response_created_events[active_speaker]
                        response_created.clear()
                        if latest_response_ids[active_speaker] != source_response_id:
                            continue
                        remaining_s = self.config.max_duration_s - (time.monotonic() - t_start)
                        if remaining_s <= 0.0:
                            stop_reason = "max_duration"
                            stop_event.set()
                            return
                        replacement_wait_s = min(
                            remaining_s,
                            max(0.0, float(self.config.silence_timeout_s)),
                        )
                        if replacement_wait_s > 0.0:
                            await _wait_for_any_event((response_created, stop_event), replacement_wait_s)
                        if stop_event.is_set():
                            return
                        if latest_response_ids[active_speaker] != source_response_id:
                            continue
                        if self.config.max_duration_s - (time.monotonic() - t_start) <= 0.0:
                            stop_reason = "max_duration"
                        else:
                            stop_reason = "response_cancelled"
                        self.event_logger.log(
                            "mode_b_source_response_cancelled",
                            speaker=active_speaker,
                            turn_index=turns_completed + 1,
                            details={
                                "response_id": source_response_id,
                                "reason": responses_by_id[source_response_key].get("cancelled_reason", "unknown"),
                                "stop_reason": stop_reason,
                            },
                        )
                        stop_event.set()
                        return
                    if not source_done.is_set():
                        conditioning_silence = np.zeros(FRAME_SAMPLES_16K, dtype=np.float32)
                        while not source_done.is_set():
                            if stop_event.is_set():
                                return
                            if source_cancelled.is_set():
                                break
                            if latest_response_ids[active_speaker] != source_response_id:
                                break
                            remaining_s = self.config.max_duration_s - (time.monotonic() - t_start)
                            if remaining_s <= 0.0:
                                stop_reason = "max_duration"
                                stop_event.set()
                                return
                            # Once response.created has been observed, silence is
                            # real no-speech ingress for the active listener: it
                            # supplies fresh V9 conditioning while native Control6
                            # maps exact-zero playback ticks to CONTINUE.
                            frame_status = await _send_input_frame(
                                active_speaker,
                                conditioning_silence,
                                source="conditioning_silence",
                                interrupt_events=(source_done, source_cancelled, stop_event),
                            )
                            if frame_status is False:
                                return
                            if frame_status is None:
                                break
                        continue

                    target_speaker = "B" if active_speaker == "A" else "A"
                    frame = self.bridge.pop_single_frame(target_speaker)
                    if frame is not None:
                        if not await _send_input_frame(target_speaker, frame, source="bridge"):
                            return
                        transfer_frames_sent += 1
                        continue

                    source_response = latest_completed_response.get(active_speaker)
                    if source_response is None or transfer_frames_sent == 0:
                        _fail_transport(
                            f"{active_speaker} completed without bridgeable output audio",
                            speaker=active_speaker,
                        )
                        return

                    created, commit_attempts = await _commit_turn(target_speaker)
                    if created is None:
                        return
                    latest_response_ids[target_speaker] = str(created.get("response_id", "")) or None
                    transfer = {
                        "from_speaker": active_speaker,
                        "to_speaker": target_speaker,
                        "source_response_id": source_response["response_id"],
                        "source_response_done_ns": source_response["response_done_ns"],
                        "target_response_id": created.get("response_id"),
                        "frames_sent": transfer_frames_sent,
                        "frames_acknowledged": transfer_frames_sent,
                        "commit_accepted": True,
                        "commit_attempts": commit_attempts,
                        "committed_at_ns": time.monotonic_ns(),
                    }
                    turn_transfers.append(transfer)
                    self.event_logger.log(
                        "turn_transfer_committed",
                        speaker=target_speaker,
                        turn_index=turns_completed + 1,
                        details=transfer,
                    )
                    active_speaker = target_speaker
                    transfer_frames_sent = 0

            task_a = asyncio.create_task(_recv_loop("A", ws_a, self.timeline_a))
            task_b = asyncio.create_task(_recv_loop("B", ws_b, self.timeline_b))

            # Session A starts the dialogue. Every 80 ms PCM frame is sent
            # only after the server confirms its listener/V9 conditioning.
            kickoff_text = scenario.steps[0].expected_text if scenario.steps else "Buongiorno!"
            kickoff_path = self.scenario_audio_dir / MODE_B_KICKOFF_FILENAME
            if kickoff_path.is_file():
                kickoff_pcm = read_scenario_audio_wav(kickoff_path)
                kickoff_audio_source = "qwen3tts_cache"
                self.event_logger.log(
                    "kickoff_audio_loaded",
                    speaker="A",
                    turn_index=1,
                    details={"path": str(kickoff_path), "samples": int(kickoff_pcm.size), "text": kickoff_text},
                )
            else:
                kickoff_pcm = generate_speech_for_text(kickoff_text)
                kickoff_audio_source = "synthetic_fallback"
                self.event_logger.log(
                    "kickoff_audio_fallback_synthetic",
                    speaker="A",
                    turn_index=1,
                    details={"samples": int(kickoff_pcm.size), "text": kickoff_text},
                )
            kickoff_frames, kickoff_remainder = pcm_to_80ms_frames(kickoff_pcm)
            if kickoff_remainder.size:
                padded = np.zeros(FRAME_SAMPLES_16K, dtype=np.float32)
                padded[:kickoff_remainder.size] = kickoff_remainder
                kickoff_frames.append(padded)
            for frame in kickoff_frames:
                if not await _send_input_frame("A", frame, source="kickoff"):
                    break
            if not stop_event.is_set():
                created, commit_attempts = await _commit_turn("A")
                if created is not None:
                    latest_response_ids["A"] = str(created.get("response_id", "")) or None
                    self.event_logger.log(
                        "kickoff_committed",
                        speaker="A",
                        turn_index=1,
                        details={"response_id": created.get("response_id"), "commit_attempts": commit_attempts},
                    )

            task_ingress = asyncio.create_task(_pacing_ingress_loop())
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.config.max_duration_s + 2.0)
            except asyncio.TimeoutError:
                stop_reason = "max_duration"
                stop_event.set()
            finally:
                stop_event.set()
                if not ingress_failures:
                    pending_ack_waits = [
                        ack_settled_events[speaker].wait()
                        for speaker in ("A", "B")
                        if inflight_ack_expected[speaker] is not None
                    ]
                    if pending_ack_waits:
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(*pending_ack_waits),
                                timeout=2.0 * self.config.ingress_ack_timeout_s + 0.1,
                            )
                        except asyncio.TimeoutError:
                            _fail_transport("listener ACK remained outstanding at shutdown")
                    late_ack_waits = [
                        abandoned_ack_drained[speaker].wait()
                        for speaker in ("A", "B")
                        if abandoned_ack_frames[speaker]
                    ]
                    late_ack_grace_s = min(
                        0.5, max(0.0, float(self.config.ingress_ack_timeout_s))
                    )
                    if late_ack_waits and late_ack_grace_s > 0.0:
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(*late_ack_waits), timeout=late_ack_grace_s
                            )
                        except asyncio.TimeoutError:
                            pass
                task_a.cancel()
                task_b.cancel()
                task_ingress.cancel()
                await asyncio.gather(task_a, task_b, task_ingress, return_exceptions=True)

            b_bridge_pcm = (
                np.concatenate(bridge_input_chunks["B"])
                if bridge_input_chunks["B"]
                else np.empty(0, dtype=np.float32)
            )
            b_input_transcript, b_input_asr_available, b_input_asr_error = await self._transcribe_input_audio(b_bridge_pcm)
            self.event_logger.log(
                "input_audio_transcribed",
                speaker="B",
                turn_index=1,
                details={
                    "source": "bridge",
                    "transcript": b_input_transcript,
                    "asr_available": b_input_asr_available,
                    **({"asr_error": b_input_asr_error} if b_input_asr_error else {}),
                },
            )
            saved_wavs = self.recorder.finalize()
            self.event_logger.close()

            pcm_out_b = self.timeline_b.get_coherent_pcm()
            response_records = sorted(responses_by_id.values(), key=lambda r: r["response_created_ns"])
            completed_responses = [r for r in response_records if r["response_done_ns"] is not None]
            turn_sequence = [str(r["speaker"]) for r in completed_responses]
            speakers_with_text_audio = {
                r["speaker"] for r in completed_responses
                if r["generated_text"].strip() and int(r["audio_samples"]) > 0
            }
            alternating = len(turn_sequence) >= 2 and {"A", "B"}.issubset(turn_sequence) and all(
                left != right for left, right in zip(turn_sequence, turn_sequence[1:])
            )
            transfers_complete = len(turn_transfers) == max(0, len(turn_sequence) - 1) and bool(turn_transfers)
            if transfers_complete:
                transfers_complete = all(
                    transfer["from_speaker"] == turn_sequence[index]
                    and transfer["to_speaker"] == turn_sequence[index + 1]
                    and transfer["frames_sent"] > 0
                    and transfer["frames_acknowledged"] == transfer["frames_sent"]
                    and transfer["commit_accepted"] is True
                    for index, transfer in enumerate(turn_transfers)
                )
            all_turn_audio_text = len(completed_responses) >= 2 and all(
                r["generated_text"].strip() and int(r["audio_samples"]) > 0
                for r in completed_responses
            )
            transport_complete = bool(alternating and transfers_complete and all_turn_audio_text)
            conversation_status = "TURN_TAKING_TRANSPORT_COMPLETE" if transport_complete else "INCOMPLETE_EVIDENCE"
            # Arrival-time overlap is not evidence of simultaneous model
            # inference. This controller intentionally serializes turns.
            overlap_pairs: List[Dict[str, Any]] = []
            interruption_records: List[Dict[str, Any]] = []
            paced_bridge_ingress_during_output: List[Dict[str, Any]] = []
            paced_bridge_duplex_observed = False
            step_eval = scenario.steps[1] if len(scenario.steps) > 1 else scenario.steps[0]
            first_a_response = next(
                (response for response in response_records if response["speaker"] == "A" and response["response_done_ns"] is not None),
                None,
            )
            b_input_reference = first_a_response["generated_text"] if first_a_response else ""
            self.evaluator.evaluate_turn(
                turn_index=1,
                speaker="B",
                scenario=scenario,
                step=step_eval,
                recognized_input=b_input_transcript,
                generated_text=accumulated_text_b,
                recognized_output=accumulated_text_b,
                audio_pcm=pcm_out_b,
                sample_rate=SAMPLE_RATE_OUTPUT,
                input_reference_text=b_input_reference,
                input_asr_available=b_input_asr_available,
            )

            summary = self.evaluator.build_summary_report()
            md_report = self.evaluator.generate_markdown_report()
            (self.output_dir / "dialogue_report.md").write_text(md_report, encoding="utf-8")
            (self.output_dir / "summary_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

            status = (
                "FAILED" if ingress_failures
                else "COMPLETED" if transport_complete and turns_completed >= self.config.max_turns
                else "INCOMPLETE"
            )
            return {
                "status": status,
                "mode": "serialized_turn_taking",
                "conversation_status": conversation_status,
                "conversation_status_basis": (
                    "completed response turns alternate A/B; each source response is fully received before its 80 ms PCM is "
                    "routed to the other session; every ingress frame has a matching listener/V9 conditioning ACK before "
                    "the recipient commit is accepted. No overlapping-inference/full-duplex claim is made."
                ),
                "stop_reason": stop_reason or ("max_turns" if turns_completed >= self.config.max_turns else "transport_stopped"),
                "turns_completed": turns_completed,
                "kickoff_audio_source": kickoff_audio_source,
                "kickoff_audio_path": str(kickoff_path) if kickoff_audio_source == "qwen3tts_cache" else None,
                "speaker_b_input_transcript": b_input_transcript,
                "turn_sequence": turn_sequence,
                "turn_transfers": turn_transfers,
                "responses": response_records,
                "input_onsets_ns": input_onsets_ns,
                "input_frames_acknowledged": dict(next_server_frame_id),
                "ingress_failures": ingress_failures,
                "audio_overlap_pairs": overlap_pairs,
                "paced_bridge_ingress_during_output": paced_bridge_ingress_during_output,
                "paced_bridge_duplex_observed": paced_bridge_duplex_observed,
                "interruptions": interruption_records,
                "report": summary,
                "markdown_report": md_report,
                "wav_files": saved_wavs,
            }


def main(argv: Optional[List[str]] = None) -> int:
    """Run a reusable real-time A or B fixture scenario against one WS session."""
    import argparse

    parser = argparse.ArgumentParser(description="Streaming single-session duplex dialogue scenarios")
    parser.add_argument("--stream-scenario", choices=("A", "B"), required=True)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("artifacts/turn_local_rollback_harness_20261018/inputs_16k"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ws-url", default="ws://localhost:8000/v1/realtime?duplex=1")
    parser.add_argument(
        "--voice-profile",
        default="gemma",
        help="Installed server voice profile for the single-session streaming scenario (default: gemma)",
    )
    parser.add_argument(
        "--turn-policy",
        default="gemma_native_control6",
        help="Server turn policy for streaming (default: gemma_native_control6)",
    )
    parser.add_argument("--max-duration-s", type=float, default=120.0)
    parser.add_argument("--silence-timeout-s", type=float, default=3.0)
    parser.add_argument("--frame-pacing-s", type=float, default=0.080)
    parser.add_argument(
        "--expected-initial-sample-count",
        type=int,
        default=None,
        help="Require this exact sample count for the first 16 kHz fixture before opening WebSocket",
    )
    args = parser.parse_args(argv)

    scenario = streaming_scenario_from_fixtures(args.stream_scenario, args.fixture_dir)
    fixture_summary = {
        "event_type": "scenario_fixture_resolved",
        "scenario_id": scenario.scenario_id,
        "steps": [
            {
                "step_id": step.step_id,
                "resolved_input_path": step.source_path,
                "sample_rate": SAMPLE_RATE_INPUT,
                "sample_count": int(step.pcm_16k.size),
            }
            for step in scenario.steps
        ],
    }
    print(json.dumps(fixture_summary, ensure_ascii=False), flush=True)
    try:
        validate_scenario_initial_sample_count(scenario, args.expected_initial_sample_count)
    except ValueError as exc:
        print(f"scenario fixture rejected before WebSocket: {exc}", file=sys.stderr, flush=True)
        return 2
    output_dir = args.output_dir or Path("artifacts") / (
        f"streaming_dialogue_{scenario.scenario_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    config = DialogueConfig(
        max_duration_s=args.max_duration_s,
        silence_timeout_s=args.silence_timeout_s,
        frame_pacing_s=args.frame_pacing_s,
        ws_url=args.ws_url,
        streaming_voice_profile=args.voice_profile,
        streaming_turn_policy=args.turn_policy,
    )
    orchestrator = DialogueOrchestrator(config=config, output_dir=output_dir)
    result = asyncio.run(orchestrator.run_streaming_scenario(scenario))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
