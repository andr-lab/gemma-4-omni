"""Opt-in same-tower parity for the standard and batched listener frontends.

Run in the configured GPU image with
``GEMMA4_RUN_LISTENER_MICROBATCH_GPU=1``. The test loads one cached Thinker,
then compares the ordinary locked ``GemmaListener.step`` path with serial
N=1 graph replays at early and full mel-history lengths. It also compares an
N=1 eager call to the reusable graph before running batches. The final LM is
skipped; compared values are real Gemma audio-tower Q4 inputs.
"""
from __future__ import annotations

from collections import deque
import os
import types
import uuid

import numpy as np
import pytest


@pytest.mark.skipif(
    os.environ.get("GEMMA4_RUN_LISTENER_MICROBATCH_GPU") != "1",
    reason="requires the configured RTX 3090 runtime and cached Thinker",
)
def test_audio_tower_q4_matches_standard_for_progressive_history_and_batch_sizes() -> None:
    import torch

    from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT
    from duplex_kernel.listener import GemmaAudioTowerBackend, GemmaListener, GemmaListenerState
    from duplex_kernel.listener_microbatch import ListenerMicrobatchCoordinator
    from duplex_kernel.listener_microbatch_frontend import ListenerMicrobatchAudioFrontend
    from duplex_kernel.microbatch_contract import (
        ListenerFrame,
        ListenerSessionState,
        NativeBatchCapability,
        RowStateUpdate,
        RuntimeSignature,
    )
    from duplex_kernel.models.thinker import load_thinker

    workspace = "/workspace"
    device = torch.device("cuda:0")
    assert torch.cuda.is_available()
    assert "3090" in torch.cuda.get_device_name(device)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    thinker = load_thinker(workspace, device=device, dtype=torch.bfloat16)
    backend = GemmaAudioTowerBackend(
        thinker,
        device=str(device),
        use_native_lm=False,
    )

    # Keep the real tower/audio frontend but avoid an unrelated LM transaction;
    # the captured input is exactly what the standard incremental LM receives.
    def capture_standard_q4(
        self,
        *,
        input_ids,
        attention_mask,
        audio_soft,
        state,
        text_ids,
    ):
        del input_ids, attention_mask, text_ids
        frame_index = len(state.user_pcm_history) - 1
        start = frame_index * 4
        return audio_soft[:, start, :], audio_soft[:, start:start + 4, :]

    backend._run_incremental_lm = types.MethodType(capture_standard_q4, backend)
    standard_listener = GemmaListener(backend=backend)
    runtime = RuntimeSignature(
        model_id="google/gemma-4-E2B-it",
        model_config_sha256="0" * 64,
        checkpoint_sha256="0" * 64,
        topology_sha256="0" * 64,
        native_library_sha256="0" * 64,
        native_abi_version=3,
        batch_api_version=2,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=4,
    )
    capability = NativeBatchCapability(
        api_version=2,
        native_abi_version=3,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=4,
        supports_arbitrary_batch_size=True,
        stream_priority=-1,
        stream_identity="listener-audio-tower-parity",
    )
    frontend = ListenerMicrobatchAudioFrontend(
        backend.gemma_model,
        backend._extract_single_frame,
        device=device,
        dtype=torch.bfloat16,
        audio_tower_lock=backend._audio_tower_workspace_lock,
        audio_tower_row_runner=backend.run_audio_tower_q4,
    )

    def pcm(session: int, frame_id: int, stream: int) -> np.ndarray:
        seed = 1_000_003 + session * 10_007 + frame_id * 101 + stream * 17
        return (np.random.default_rng(seed).standard_normal(1280) * 0.05).astype(np.float32)

    exact_mismatches: list[str] = []
    for n in (1, 2, 4, 8, 12):
        for history_frames in (1, 2, 63, 64):
            run_id = uuid.uuid4().hex
            coordinator = ListenerMicrobatchCoordinator(
                runtime, capability, coalesce_window_ns=0
            )
            standard_states: list[GemmaListenerState] = []
            batch_states: list[ListenerSessionState] = []
            frames: list[ListenerFrame] = []
            current_pcm: list[np.ndarray] = []
            current_reference: list[np.ndarray | None] = []
            standard_feature_inputs: list[torch.Tensor] = []
            standard_feature_masks: list[torch.Tensor] = []

            for row in range(n):
                session_id = f"{run_id}-{row}"
                mel_user: list[np.ndarray] = []
                mel_assistant: list[np.ndarray] = []
                user_tail: np.ndarray | None = None
                assistant_tail: np.ndarray | None = None
                prior_user_pcm: list[np.ndarray] = []
                prior_assistant_pcm: list[np.ndarray] = []
                for prior_frame in range(history_frames - 1):
                    user_wave = pcm(row, prior_frame, 0)
                    assistant_wave = (
                        np.zeros(1280, dtype=np.float32)
                        if prior_frame == 0
                        else pcm(row, prior_frame - 1, 1)
                    )
                    user_mel, user_tail = backend._extract_single_frame(user_wave, user_tail)
                    assistant_mel, assistant_tail = backend._extract_single_frame(
                        assistant_wave, assistant_tail
                    )
                    prior_user_pcm.append(user_wave)
                    prior_assistant_pcm.append(assistant_wave)
                    mel_user.append(user_mel)
                    mel_assistant.append(assistant_mel)

                frame_id = history_frames - 1
                user_wave = pcm(row, frame_id, 0)
                reference_wave = None if frame_id == 0 else pcm(row, frame_id - 1, 1)
                current_pcm.append(user_wave)
                current_reference.append(reference_wave)

                standard = GemmaListenerState(next_frame_id=frame_id)
                standard.user_pcm_history.extend(prior_user_pcm)
                standard.assistant_pcm_history.extend(prior_assistant_pcm)
                standard.user_mel_history.extend(mel_user)
                standard.assistant_mel_history.extend(mel_assistant)
                standard.user_pcm_tail = user_tail
                standard.assistant_pcm_tail = assistant_tail
                standard_states.append(standard)

                batch_state = ListenerSessionState(
                    session_id=session_id,
                    generation_id=1,
                    state_epoch_id=f"epoch-{session_id}",
                    runtime_signature=runtime,
                    next_frame_id=frame_id,
                    state_version=frame_id,
                    sequence_position=frame_id,
                    cache_position=AUDIO_START_AFTER_TEXT + 4 * frame_id,
                    valid_history_tokens=AUDIO_START_AFTER_TEXT + 4 * frame_id,
                    native_context_id=100_000 + n * 100 + row,
                    user_pcm_tail=user_tail,
                    assistant_pcm_tail=assistant_tail,
                    user_mel_history=deque(mel_user, maxlen=64),
                    assistant_mel_history=deque(mel_assistant, maxlen=64),
                )
                coordinator.register_session(batch_state)
                batch_states.append(batch_state)
                frames.append(ListenerFrame(
                    session_id=session_id,
                    generation_id=1,
                    state_epoch_id=f"epoch-{session_id}",
                    frame_id=frame_id,
                    ready_at_ns=0,
                    pcm=user_wave,
                    reference_pcm=reference_wave,
                    reference_frame_id=None if reference_wave is None else frame_id - 1,
                ))

            expected_q4: list[torch.Tensor] = []
            for user_wave, reference_wave, standard in zip(
                current_pcm, current_reference, standard_states, strict=True
            ):
                output = standard_listener.step(user_wave, reference_wave, standard)
                full_standard_tokens = output.audio_tower_hidden
                assert isinstance(full_standard_tokens, torch.Tensor)
                assert backend._static_features is not None
                assert backend._static_mask is not None
                standard_feature_inputs.append(backend._static_features.detach().clone())
                standard_feature_masks.append(backend._static_mask.detach().clone())
                frame_start = (history_frames - 1) * 4
                expected_q4.append(
                    full_standard_tokens[frame_start:frame_start + 4].detach().clone()
                )

            if n == 1 and history_frames == 1:
                probe_features = standard_feature_inputs[0]
                probe_mask = torch.ones((1, 1024), dtype=torch.bool, device=device)
                with torch.inference_mode():
                    eager_output = backend.gemma_model.get_audio_features(
                        input_features=probe_features,
                        input_features_mask=probe_mask,
                        return_dict=True,
                    )
                eager_valid = eager_output.pooler_output[0][
                    eager_output.attention_mask[0].to(dtype=torch.bool)
                ]
                eager_q4 = eager_valid[:4].detach().clone()
                graph_q4, graph_timing = backend.run_audio_tower_q4(probe_features, 0)
                torch.cuda.synchronize(device)
                eager_graph_exact = torch.equal(eager_q4, graph_q4[0])
                print(
                    "[frontend-n1-eager-graph] "
                    f"exact={eager_graph_exact} mode={graph_timing.execution_mode} "
                    f"max_abs={float((eager_q4.float() - graph_q4[0].float()).abs().max().item()):.8g}",
                    flush=True,
                )
                assert eager_graph_exact, "N=1 eager tower and captured graph differ"

            for frame in frames:
                coordinator.enqueue(frame)
            batch = coordinator.take_ready(1)
            assert batch is not None and batch.manifest.batch_size == n
            prepared = frontend.encode_batch(batch, batch_states)
            assert frontend._workspace_buffer is not None
            actual_features = frontend._workspace_buffer.features[:n].detach().clone()
            actual_mask = frontend._workspace_buffer.feature_mask[:n].detach().clone()
            expected_features = torch.cat(standard_feature_inputs, dim=0)
            expected_mask = torch.cat(standard_feature_masks, dim=0)
            feature_difference = (actual_features.float() - expected_features.float()).abs()
            feature_inputs_exact = torch.equal(actual_features, expected_features)
            masks_equal = torch.equal(actual_mask, expected_mask)
            actual_batch = torch.stack(prepared.audio_soft_tokens, dim=0).float()
            expected_batch = torch.stack(expected_q4, dim=0).float()
            difference = (actual_batch - expected_batch).abs()
            safe_denominator = expected_batch.abs().clamp_min(1e-3)
            relative = difference / safe_denominator
            exact = torch.equal(actual_batch, expected_batch)
            first_difference = (
                None
                if exact
                else tuple(
                    int(value)
                    for value in torch.nonzero(
                        actual_batch != expected_batch, as_tuple=False
                    )[0].tolist()
                )
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    actual_batch.reshape(1, -1), expected_batch.reshape(1, -1), dim=1
                )[0].item()
            )
            print(
                "[frontend-q4] "
                f"N={n} history={history_frames} exact={exact} "
                f"feature_inputs_exact={feature_inputs_exact} "
                f"feature_input_max_abs={float(feature_difference.max().item()):.8g} "
                f"masks_equal={masks_equal} "
                f"max_abs={float(difference.max().item()):.8g} "
                f"max_rel_floor_1e-3={float(relative.max().item()):.8g} "
                f"rms_abs={float(difference.square().mean().sqrt().item()):.8g} "
                f"rms_rel_floor_1e-3={float(relative.square().mean().sqrt().item()):.8g} "
                f"cosine={cosine:.10g} first_difference={first_difference}",
                flush=True,
            )
            if not exact:
                exact_mismatches.append(
                    f"N={n}, history={history_frames}, first={first_difference}, "
                    f"max_abs={float(difference.max().item()):.8g}"
                )
            if not feature_inputs_exact or not masks_equal:
                exact_mismatches.append(
                    f"input mismatch N={n}, history={history_frames}, "
                    f"feature_max_abs={float(feature_difference.max().item()):.8g}, "
                    f"masks_equal={masks_equal}"
                )
            prepared.commit()
            coordinator.complete(
                batch,
                tuple(
                    RowStateUpdate(
                        session_id=row.session_id,
                        frame_id=row.frame_id,
                        cache_position=row.cache_position + 4,
                        valid_history_tokens=row.valid_history_tokens + 4,
                    )
                    for row in batch.manifest.rows
                ),
            )
            del batch, prepared, batch_states, standard_states
            torch.cuda.synchronize(device)

    assert not exact_mismatches, (
        "audio-tower Q4 differs from the exact standard output; see per-case metrics above: "
        + "; ".join(exact_mismatches)
    )

