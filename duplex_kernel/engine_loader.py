from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2
from duplex_kernel.config import SPEAKER_REFERENCE_SHA256, SPEAKER_REFERENCE_WAV, V9_PROFILE_CAUSAL_L0
from duplex_kernel.engine_prefill import _build_backchannel_token_trie, _validate_response_bundle_qlora_compatibility
from duplex_kernel.engine_talk_context_pool import initialize_response_talk_context_pool
from duplex_kernel.frame_residual_adapter import load_frame_residual_adapter
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.thinker import load_thinker, normalize_eos_token_ids
from duplex_kernel.models.v9 import IncrementalV9Projector, load_v9_conditioning

logger = logging.getLogger("duplex_kernel.engine")
_TALKER_COMPILED_DECODE_ENV = "DUPLEX_TALKER_COMPILED_DECODE"
_TALKER_NATIVE_AUDIO_Q1_ENV = "DUPLEX_TALKER_NATIVE_AUDIO_Q1"


def load_engine_resources(
    engine: Any,
    *,
    workspace: str = "/workspace",
    speaker_reference_wav: str | None = None,
    speaker_reference_sha256: str | None = None,
) -> bool:
    try:
        t0 = time.time()
        engine.workspace = Path(workspace)
        configured_control6_paths = (
            engine.duplex_control6_adapter_dir,
            engine.duplex_control6_head_path,
            engine.duplex_control6_manifest_path,
        )

        def workspace_path(value: str | None) -> str | None:
            if value is None:
                return None
            candidate = Path(value)
            return str(candidate if candidate.is_absolute() else Path(workspace) / candidate)

        if engine.enable_gemma_native_control6_policy:
            if not engine.native_control6_adapter_path:
                raise ValueError("gemma_native_control6 requires --native-control6-adapter")
            if not engine.ablate_response_lora and (not engine.gemma_qlora_adapter_dir):
                raise ValueError("gemma_native_control6 requires response LoRA disabled")
            engine.native_control6_adapter_path = workspace_path(engine.native_control6_adapter_path)
            if not Path(engine.native_control6_adapter_path).is_file():
                raise FileNotFoundError(f"native Control6 checkpoint missing: {engine.native_control6_adapter_path}")
            if engine.native_control6_semantic_prefix_enabled:
                from duplex_kernel.native_control6_prompt import (
                    validate_native_control6_semantic_prefix_metadata,
                )

                checkpoint_metadata = torch.load(
                    engine.native_control6_adapter_path,
                    map_location="cpu",
                    weights_only=True,
                )
                engine.native_control6_semantic_prefix_metadata = validate_native_control6_semantic_prefix_metadata(checkpoint_metadata)
        elif engine.native_control6_adapter_path is not None:
            raise ValueError("--native-control6-adapter requires --enable-gemma-native-control6")
        if engine.ablate_response_lora and (not engine.duplex_response_bundle_dir) and (not engine.enable_gemma_native_control6_policy):
            raise ValueError("--ablate-response-lora richiede --duplex-response-bundle-dir")
        if engine.duplex_response_bundle_dir:
            if any(configured_control6_paths):
                raise ValueError("il response bundle include il proprio adapter/head; non combinarlo con asset legacy")
            _validate_response_bundle_qlora_compatibility(
                bundle_enabled=True,
                gemma_qlora_enabled=bool(engine.gemma_qlora_adapter_dir),
                native_control6_enabled=engine.enable_gemma_native_control6_policy,
                response_lora_ablated=engine.ablate_response_lora,
            )
            engine.duplex_response_bundle_dir = workspace_path(engine.duplex_response_bundle_dir)
            bundle_root = Path(engine.duplex_response_bundle_dir)
            if not bundle_root.is_dir() or not (bundle_root / "bundle_manifest.json").is_file():
                raise FileNotFoundError(f"duplex response bundle incompleto: {bundle_root}/bundle_manifest.json")
            from duplex_kernel.audio_response_protocol import (
                normalize_audio_response_protocol,
            )
            from duplex_kernel.duplex_control6 import read_duplex_response_bundle

            bundle_contract = read_duplex_response_bundle(engine.duplex_response_bundle_dir)
            engine.audio_response_protocol = normalize_audio_response_protocol(bundle_contract["manifest"].get("response_protocol"))
        elif engine.enable_duplex_control6:
            if not engine.duplex_control6_adapter_dir or not engine.duplex_control6_manifest_path:
                raise ValueError("--enable-duplex-control6 richiede adapter LoRA e training manifest; la policy non può fare fallback silenzioso")
            if engine.gemma_qlora_adapter_dir:
                raise ValueError("specificare --gemma-qlora-adapter-dir per il solo smoke oppure --enable-duplex-control6 per policy+adapter, non entrambi")
            engine.duplex_control6_adapter_dir = workspace_path(engine.duplex_control6_adapter_dir)
            engine.duplex_control6_head_path = workspace_path(engine.duplex_control6_head_path)
            engine.duplex_control6_manifest_path = workspace_path(engine.duplex_control6_manifest_path)
            if not Path(engine.duplex_control6_adapter_dir).is_dir():
                raise FileNotFoundError(f"duplex_control6 adapter mancante: {engine.duplex_control6_adapter_dir}")
            if not Path(engine.duplex_control6_manifest_path).is_file():
                raise FileNotFoundError(f"duplex_control6 manifest mancante: {engine.duplex_control6_manifest_path}")
        elif any(configured_control6_paths):
            raise ValueError("asset duplex_control6 richiedono --enable-duplex-control6 esplicito")
        if engine._talker_native_audio_q1_requested and engine.audio_response_protocol != AUDIO_RESPONSE_PROTOCOL_V2:
            raise ValueError(f"{_TALKER_NATIVE_AUDIO_Q1_ENV}=1 requires {AUDIO_RESPONSE_PROTOCOL_V2}")
        if engine.gemma_qlora_adapter_dir:
            candidate = Path(engine.gemma_qlora_adapter_dir)
            engine.gemma_qlora_adapter_dir = str(candidate if candidate.is_absolute() else Path(workspace) / candidate)
            if not Path(engine.gemma_qlora_adapter_dir).is_dir():
                raise FileNotFoundError(f"Gemma QLoRA adapter mancante: {engine.gemma_qlora_adapter_dir}")
        th = load_thinker(workspace, device=engine.device, dtype=engine.dtype)
        engine._thinker_wrapper = th
        if engine.duplex_response_bundle_dir:
            from duplex_kernel.duplex_control6 import (
                load_duplex_response_bundle_runtime,
            )

            engine.duplex_control6_runtime = load_duplex_response_bundle_runtime(
                th,
                bundle_dir=engine.duplex_response_bundle_dir,
                device=engine.device,
                apply_response_lora=not engine.ablate_response_lora,
            )
            logger.info(
                "[engine] response bundle merged before listener binding bundle_id=%s adapter=%s head=%s protocol=%s",
                engine.duplex_control6_runtime["bundle_id"],
                engine.duplex_control6_runtime["adapter_path"],
                engine.duplex_control6_runtime["control_head"],
                engine.audio_response_protocol or "legacy_v1",
            )
        elif engine.enable_duplex_control6:
            from duplex_kernel.duplex_control6 import load_control6_runtime

            engine.duplex_control6_runtime = load_control6_runtime(
                th,
                adapter_dir=engine.duplex_control6_adapter_dir,
                head_path=engine.duplex_control6_head_path,
                manifest_path=engine.duplex_control6_manifest_path,
                device=engine.device,
            )
            logger.info(
                "[engine] duplex_control6 merged before listener pointer binding run_id=%s base=%s adapter=%s",
                engine.duplex_control6_runtime["run_id"],
                engine.duplex_control6_runtime["base_path"],
                engine.duplex_control6_runtime["adapter_path"],
            )
        if engine.gemma_qlora_adapter_dir:
            from duplex_kernel.duplex_control6 import merge_gemma_adapter_into_thinker

            base_path = merge_gemma_adapter_into_thinker(th, engine.gemma_qlora_adapter_dir)
            engine.gemma_qlora_adapter_loaded = str(Path(engine.gemma_qlora_adapter_dir).resolve())
            logger.info(
                "[engine] Gemma QLoRA adapter-only merged before listener pointer binding base=%s adapter=%s",
                base_path,
                engine.gemma_qlora_adapter_loaded,
            )
        if engine._talker_compiled_decode_requested and engine.audio_response_protocol != AUDIO_RESPONSE_PROTOCOL_V2:
            raise ValueError(f"{_TALKER_COMPILED_DECODE_ENV}=1 requires response protocol {AUDIO_RESPONSE_PROTOCOL_V2}")
        engine.thinker = th.model
        engine.tokenizer = th.tokenizer
        if engine.enable_gemma_native_control6_policy:
            if engine.native_control6_semantic_prefix_enabled:
                live_boa_token_id = getattr(getattr(engine.thinker, "config", None), "boa_token_id", None)
                checkpoint_boa_token_id = engine.native_control6_semantic_prefix_metadata["semantic_prefix_boa_token_id"]
                if live_boa_token_id is None or int(live_boa_token_id) != int(checkpoint_boa_token_id):
                    raise RuntimeError(f"native Control6 checkpoint BOA token differs from the loaded Gemma config: checkpoint={checkpoint_boa_token_id!r}, live={live_boa_token_id!r}")
            from training.duplex_qlora.native_control6 import (
                load_native_control6_adapter,
                select_tied_output_rows,
            )

            output_head = getattr(engine.thinker, "lm_head", None)
            output_weight = getattr(output_head, "weight", None)
            if output_weight is None:
                raise RuntimeError("loaded Gemma thinker has no native lm_head.weight")
            live_rows = select_tied_output_rows(output_weight)
            engine.native_control6_runtime, engine.native_control6_report = load_native_control6_adapter(
                engine.native_control6_adapter_path,
                base_rows=live_rows,
                device=engine.device,
            )
            engine.native_control6_runtime.eval()
            for parameter in engine.native_control6_runtime.parameters():
                parameter.requires_grad_(False)
            logger.info(
                "[engine] native Gemma Control6 loaded checkpoint=%s rows=6 hidden=1536 rank=%d response_lora_applied=false",
                engine.native_control6_adapter_path,
                engine.native_control6_runtime.rank,
            )
        if engine.enable_duplex_control6:
            runtime_lexicon_path = (engine.duplex_control6_runtime or {}).get("backchannel_lexicon_path")
            lexicon_path = Path(runtime_lexicon_path) if runtime_lexicon_path else Path(engine.duplex_control6_manifest_path).with_name("backchannel_lexicon.json")
            if not lexicon_path.is_file():
                raise FileNotFoundError(f"backchannel lexicon train mancante nel bundle attivo: {lexicon_path}")
            with lexicon_path.open("r", encoding="utf-8") as handle:
                lexicon = json.load(handle)
            if not isinstance(lexicon, dict) or lexicon.get("schema_version") != 1:
                raise ValueError("schema backchannel_lexicon.json non supportato")
            if lexicon.get("language") != "it":
                raise ValueError("backchannel_lexicon.json deve dichiarare language='it'")
            if lexicon.get("source_split", "train") != "train":
                raise ValueError("backchannel_lexicon.json deve provenire dallo split train")
            phrases = lexicon.get("phrases")
            if not isinstance(phrases, list) or any((not isinstance(phrase, str) for phrase in phrases)):
                raise ValueError("backchannel_lexicon.json deve contenere phrases: list[str]")
            eos_value = getattr(getattr(engine.thinker, "generation_config", None), "eos_token_id", None)
            if eos_value is None:
                eos_value = getattr(getattr(engine.thinker, "config", None), "eos_token_id", None)
            eos_ids = normalize_eos_token_ids(eos_value)
            if not eos_ids:
                eos_ids = normalize_eos_token_ids(getattr(engine.tokenizer, "eos_token_id", None))
                eos_ids.update(normalize_eos_token_ids(getattr(engine.tokenizer, "eot_id", None)))
            engine._backchannel_token_trie = _build_backchannel_token_trie(phrases, engine.tokenizer, eos_token_ids=eos_ids)
            engine._backchannel_say_phrases = tuple(dict.fromkeys((phrase.strip() for phrase in phrases)))
            engine._backchannel_phrase_count = len(engine._backchannel_say_phrases)
            logger.info(
                "[engine] BACKCHANNEL constrained to %d train phrases / %d token prefixes from %s",
                engine._backchannel_phrase_count,
                len(engine._backchannel_token_trie),
                lexicon_path,
            )
        from duplex_kernel.listener import GemmaListener

        engine.listener = GemmaListener.from_thinker(
            th,
            runtime_profile="production",
            enable_silence_skip=engine.enable_listener_silence_skip,
        )
        if hasattr(engine.listener, "backend") and hasattr(engine.listener.backend, "_lock"):
            engine._thinker_lock = engine.listener.backend._lock
        logger.info(
            "[engine] thinker + continuous listener in %.1fs | prefill_chunk_tokens=%d audio_lm_chunk_tokens=%d compile=%s static_cache=%d",
            time.time() - t0,
            engine.thinker_prefill_chunk_tokens,
            engine.thinker_audio_prefill_chunk_tokens,
            engine.thinker_compile_mode,
            engine.thinker_static_cache_max_len,
        )
        if engine.ablate_session_voice_conditioning:
            resolved_speaker_wav = None
            resolved_speaker_sha = None
        else:
            resolved_speaker_wav = speaker_reference_wav or SPEAKER_REFERENCE_WAV or None
            resolved_speaker_sha = speaker_reference_sha256 or SPEAKER_REFERENCE_SHA256 or None
        if (resolved_speaker_wav is None) != (resolved_speaker_sha is None):
            raise ValueError("speaker reference path/SHA incompleti")
        engine.talker = load_talker(
            workspace,
            device=engine.device,
            speaker_reference_wav=resolved_speaker_wav,
            speaker_reference_sha256=resolved_speaker_sha,
        )
        logger.info(
            "[engine] talker+code2wav recipe=%s in %.1fs",
            engine.qwen_audio_recipe,
            time.time() - t0,
        )
        engine.v9, engine.causal_conv = load_v9_conditioning(profile=engine.v9_profile, device=engine.device, dtype=engine.dtype)
        engine.v9_recipe = getattr(engine.v9, "_v9_recipe", None)
        if engine.v9_recipe is None:
            raise RuntimeError("V9 loader non ha attestato la recipe")
        engine.v9_stream = IncrementalV9Projector(engine.v9, engine.causal_conv, max_seq_len=16384) if engine.v9_profile == V9_PROFILE_CAUSAL_L0 else None
        logger.info(
            "[engine] V9 profile=%s recipe=%s in %.1fs",
            engine.v9_profile,
            engine.v9_recipe,
            time.time() - t0,
        )
        initialize_response_talk_context_pool(engine)
        if engine.thinker_compile_mode != "off":
            engine._prepare_compiled_thinker()
        if engine._talker_compiled_decode_requested and (not engine._talker_native_audio_q1_requested):
            engine._prepare_compiled_talker_q1()
        if engine.frame_residual_adapter_checkpoint_path is not None:
            if engine.runtime_backend != "native":
                raise ValueError("frame residual adapter requires runtime_backend=native")
            if engine.audio_response_protocol != AUDIO_RESPONSE_PROTOCOL_V2:
                raise ValueError("frame residual adapter requires heard_action_say_v2 direct audio")
            if not engine.duplex_response_bundle_dir or not engine.ablate_response_lora:
                raise ValueError("frame residual adapter requires controller bundle with response LoRA ablated")
            if int(engine.controller_guided_talk_contexts) < 1:
                raise ValueError("frame residual adapter requires a positive response pool")
            if not bool(getattr(engine, "enforce_v9_training_voice", False)) or engine.ablate_session_voice_conditioning:
                raise ValueError("frame residual adapter requires the exact training-default speaker conditioning")
            if engine.v9_conditioning_source != "live_generated" or engine.v9_dual_audio_context_experimental:
                raise ValueError("frame residual adapter requires the plain semantic-text Qwen route")
            if getattr(engine, "response_intent_local_projector", None) is not None:
                raise ValueError("frame residual adapter cannot share a response with a prefix projector")
            if not engine._talker_native_audio_q1_requested:
                raise ValueError("frame residual adapter requires requested native audio-Q1")
            adapter_path = Path(engine.frame_residual_adapter_checkpoint_path).expanduser()
            if not adapter_path.is_absolute():
                adapter_path = engine.workspace / adapter_path
            engine.frame_residual_adapter, payload = load_frame_residual_adapter(adapter_path, device=engine.device, dtype=engine.dtype)
            if engine.frame_residual_adapter.architecture_id != "response_hidden_q1_query_film_v2":
                raise ValueError("production serving requires the response_hidden_q1_query_film_v2 Query FiLM v2 checkpoint")
            engine.frame_residual_adapter_resolved_path = str(adapter_path.resolve())
            engine.frame_residual_adapter_architecture = str(engine.frame_residual_adapter.architecture_id)
            logger.info(
                "[engine] frame_residual_q1 loaded architecture=%s schema=%s checkpoint=%s route=plain_native_q1 frozen=true",
                engine.frame_residual_adapter_architecture,
                payload.get("schema_version"),
                engine.frame_residual_adapter_resolved_path,
            )
        if engine.audio_response_protocol == "heard_action_say_v2":
            engine._warmup_direct_v2_audio_path()
        if engine.ablate_session_voice_conditioning:
            engine.talker.default_voice_conditioning = None
            if hasattr(engine.talker, "voice_conditioning"):
                delattr(engine.talker, "voice_conditioning")
            logger.info("[engine] session voice-conditioning ablation enabled; no speaker reference or profile will be used for sessions")
        if engine.diagnostic_talker_seed is not None:
            import random

            random.seed(engine.diagnostic_talker_seed)
            np.random.seed(engine.diagnostic_talker_seed)
            torch.manual_seed(engine.diagnostic_talker_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(engine.diagnostic_talker_seed)
            logger.info(
                "[engine] diagnostic_talker_seed=%d scope=startup_global_not_per_response",
                engine.diagnostic_talker_seed,
            )
        engine.loaded = True
        logger.info("[engine] LOAD COMPLETO in %.1fs", time.time() - t0)
        return True
    except Exception as exc:
        logger.exception("Engine load fallito: %s", exc)
        return False
