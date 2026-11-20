from __future__ import annotations

import json
import logging
import time
from typing import Any

import torch

from duplex_kernel import config
from duplex_kernel.audio_stream import CODE2WAV_DECODE_FRAMES
from duplex_kernel.response_talk_context import ResponseTalkContextPool
from duplex_kernel.speaker_runtime import Code2WavWindowGraph, Code2WavWindowGraphFamily
from duplex_kernel.subtalker_graph import SubtalkerGraph

logger = logging.getLogger("duplex_kernel.engine")


def initialize_response_talk_context_pool(engine: Any) -> None:
    """Build the shared Talker graph pool and its memory diagnostics."""
    cp = engine.talker.talker.code_predictor
    engine.subtalker_graph = SubtalkerGraph(
        cp,
        talker_embed=engine.talker.talker.get_input_embeddings(),
        device=engine.device,
    ).capture()
    logger.info(
        "[engine] subtalker graph: %s",
        "catturato" if engine.subtalker_graph.captured else "eager",
    )
    if not engine.controller_guided_talk_contexts:
        return

    if torch.cuda.is_available() and str(engine.device).startswith("cuda"):
        torch.cuda.synchronize(device=engine.device)
        allocated_before = int(torch.cuda.memory_allocated(device=engine.device))
        reserved_before = int(torch.cuda.memory_reserved(device=engine.device))
    else:
        allocated_before = reserved_before = 0
    talk_graphs = [engine.subtalker_graph]
    for _slot_index in range(1, engine.controller_guided_talk_contexts):
        talk_graphs.append(SubtalkerGraph(
            cp,
            talker_embed=engine.talker.talker.get_input_embeddings(),
            device=engine.device,
        ).capture())

    code2wav_graphs = None
    code2wav_graph_prerequisites = (
        torch.cuda.is_available()
        and str(engine.device).startswith("cuda")
        and CODE2WAV_DECODE_FRAMES == Code2WavWindowGraph.NEW_FRAMES
        and getattr(engine.talker, "code2wav", None) is not None
    )
    if config.CODE2WAV_GRAPH_FAMILY_ENABLED and not code2wav_graph_prerequisites:
        raise RuntimeError(
            "requested Code2Wav graph family requires CUDA, CODE2WAV_DECODE_FRAMES=2, "
            "and a loaded Code2Wav decoder"
        )
    if config.CODE2WAV_GRAPH_FAMILY_ENABLED:
        code2wav_graphs = []
        family_capture_started = time.perf_counter()
        for _slot_index in range(engine.controller_guided_talk_contexts):
            code2wav_graphs.append(Code2WavWindowGraphFamily(
                engine.talker.code2wav,
                device=str(engine.device),
                include_odd_tail=config.CODE2WAV_GRAPH_ODD_TAIL_ENABLED,
            ).capture())
        family_capture_seconds = time.perf_counter() - family_capture_started
        logger.info(
            "[engine] Code2Wav exact-window graph family captured slots=%d/%d "
            "graphs_per_slot=%d odd_tail=%s windows=%s capture_seconds=%.3f "
            "static_buffers_bytes_per_slot=%d",
            sum(family.captured for family in code2wav_graphs),
            len(code2wav_graphs),
            len(code2wav_graphs[0].window_frames) if code2wav_graphs else 0,
            config.CODE2WAV_GRAPH_ODD_TAIL_ENABLED,
            list(code2wav_graphs[0].window_frames) if code2wav_graphs else [],
            family_capture_seconds,
            code2wav_graphs[0].static_buffers_bytes if code2wav_graphs else 0,
        )
    elif code2wav_graph_prerequisites and config.CODE2WAV_GRAPH_ENABLED:
        code2wav_graphs = []
        for _slot_index in range(engine.controller_guided_talk_contexts):
            decode_graph = Code2WavWindowGraph(
                engine.talker.code2wav, device=str(engine.device)
            )
            decode_stream = torch.cuda.Stream(device=torch.device(engine.device), priority=0)
            decode_graph.capture(decode_stream)
            decode_stream.synchronize()
            code2wav_graphs.append(decode_graph)
        logger.info(
            "[engine] Code2Wav window graphs catturati: %d/%d",
            sum(graph.captured for graph in code2wav_graphs),
            len(code2wav_graphs),
        )

    engine.response_talk_context_pool = ResponseTalkContextPool(
        talk_graphs,
        code2wav_graphs=code2wav_graphs,
        admission_capacity=int(engine.controller_guided_talk_contexts),
    )
    element_bytes = int(torch.tensor([], dtype=engine.dtype).element_size())
    cp_config = getattr(cp, "config", None)
    cp_layers = int(getattr(cp_config, "num_hidden_layers", 0) or 0)
    cp_heads = int(getattr(cp_config, "num_key_value_heads", 0) or getattr(cp_config, "num_attention_heads", 0) or 0)
    cp_hidden = int(getattr(cp_config, "hidden_size", 0) or 0)
    cp_head_dim = int(getattr(cp_config, "head_dim", 0) or (cp_hidden // cp_heads if cp_heads else 0))
    qwen_config = getattr(engine.talker.talker, "config", None)
    qwen_layers = int(getattr(qwen_config, "num_hidden_layers", 0) or 0)
    qwen_heads = int(getattr(qwen_config, "num_key_value_heads", 0) or getattr(qwen_config, "num_attention_heads", 0) or 0)
    qwen_hidden = int(getattr(qwen_config, "hidden_size", 0) or 0)
    qwen_head_dim = int(getattr(qwen_config, "head_dim", 0) or (qwen_hidden // qwen_heads if qwen_heads else 0))
    if torch.cuda.is_available() and str(engine.device).startswith("cuda"):
        torch.cuda.synchronize(device=engine.device)
        allocated_after = int(torch.cuda.memory_allocated(device=engine.device))
        reserved_after = int(torch.cuda.memory_reserved(device=engine.device))
    else:
        allocated_after = reserved_after = 0
    graph_infos = [graph.runtime_info() for graph in code2wav_graphs] if code2wav_graphs is not None else []
    engine.response_talk_context_memory = {
        "capacity": int(getattr(engine.response_talk_context_pool, "admission_capacity", len(talk_graphs))),
        "allocated_before_bytes": allocated_before,
        "allocated_after_bytes": allocated_after,
        "allocated_delta_bytes": allocated_after - allocated_before,
        "reserved_before_bytes": reserved_before,
        "reserved_after_bytes": reserved_after,
        "reserved_delta_bytes": reserved_after - reserved_before,
        "subtalker_static_kv_estimate_bytes_per_slot": cp_layers * 2 * cp_heads * cp_head_dim * (engine.subtalker_graph.max_code_groups + 2) * element_bytes,
        "qwen_kv_estimate_bytes_per_token": qwen_layers * 2 * qwen_heads * qwen_head_dim * element_bytes,
        "code2wav_graph_family_enabled": bool(config.CODE2WAV_GRAPH_FAMILY_ENABLED),
        "code2wav_graph_odd_tail_enabled": bool(config.CODE2WAV_GRAPH_ODD_TAIL_ENABLED),
        "code2wav_graph_capture_status_by_slot": [
            {
                "captured": bool(info.get("captured")),
                "graph_count": int(info.get("graph_count", 1 if info.get("captured") else 0)),
                "window_frames": info.get("window_frames"),
                "static_buffers_bytes": int(info.get("static_buffers_bytes", 0)),
                "capture_error": info.get("capture_error"),
            }
            for info in graph_infos
        ],
    }
    logger.info(
        "[engine] response_talk_context_pool %s",
        json.dumps(engine.response_talk_context_memory, sort_keys=True),
    )
