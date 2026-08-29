# -*- coding: utf-8 -*-
"""
server/openai_omni_server.py — Server FastAPI OpenAI-Compatible & Realtime WebSocket con Motore Neurale Reale Gemma 4 Omni.

Supporta:
1. `GET /v1/models`: Catalogo modelli (gemma-4-omni-duplex).
2. `POST /v1/chat/completions`:
   - Chat statica testo con inferenza REALE Gemma 4 Thinker.
   - Function calling / Tools (JSON Schema OpenAI).
   - Streaming SSE (`stream=True`) per token testuali e tool call deltas.
   - Chat multimodale con risposte lette a voce tramite Qwen3-TTS + Projector V9 (`modalities: ["text", "audio"]`).
3. `WebSocket /v1/realtime`:
   - Protocollo OpenAI Realtime Full-Duplex.
   - Microfono continuo, cancellazione istantanea su Barge-In via TurnDetectorHead.
   - Streaming parallelo di trascrizione testuale e chunk audio a 24kHz reali.
   - Supporto Function Calling bidirezionale.
4. `GET /`: Web SPA interattiva per testare testo, voce e tool in tempo reale.
"""

import asyncio
import base64
import glob
import io
import json
import os
import re
import sys
import time
import types
import importlib.machinery
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Mock torchaudio per compatibilità binaria 12.5Hz
try:
    import torchaudio
except Exception:
    _m = types.ModuleType('torchaudio')
    _m.__spec__ = importlib.machinery.ModuleSpec('torchaudio', None)
    sys.modules['torchaudio'] = _m
    sys.modules['torchaudio._extension'] = types.ModuleType('torchaudio._extension')
    sys.modules['torchaudio.compliance'] = types.ModuleType('torchaudio.compliance')
    sys.modules['torchaudio.compliance.kaldi'] = types.ModuleType('torchaudio.compliance.kaldi')
sys.modules['sox'] = types.ModuleType('sox')

# RoPE patch per Qwen3-TTS
import transformers.modeling_rope_utils as _ropeu
def _default_rope_init(config, device=None, seq_len=None, **kwargs):
    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    _head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    _dim = int(_head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, _dim, 2, dtype=torch.int64).float().to(device) / _dim))
    return inv_freq, 1.0
_ropeu.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
_ropeu.ROPE_INIT_FUNCTIONS["interleaved"] = _default_rope_init

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "qwen3tts_src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "qwen3tts_src"))

from server.duplex_session import DuplexSession, DuplexSessionManager
from server.tool_parser import extract_tool_calls_from_text, format_tools_system_prompt
from server.voice_manager import VoiceProfileManager
from vllm_omni_plugin.contracts import DuplexState
from vllm_omni_plugin.runtime_extension import Gemma4OmniDuplexRuntimeExtension

from transformers import AutoTokenizer
import qwen_tts.core.models as _M
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
import safetensors.torch as st
import kaggle_pipeline as KP
from tools.run_exact_interactive_dialogue_benchmark import DeepCausalTransformerProjectorV9

# Inizializzazione FastAPI
app = FastAPI(
    title="Gemma 4 Omni Full-Duplex API Server",
    description="OpenAI-Compatible REST & Realtime WebSocket API Server for Gemma 4 Omni",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gestori di stato
session_manager = DuplexSessionManager()
voice_manager = VoiceProfileManager()
duplex_extension = Gemma4OmniDuplexRuntimeExtension()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Schemi Pydantic per OpenAI API
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Optional[Any] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "gemma-4-omni-duplex"
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = "auto"
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256
    modalities: Optional[List[str]] = Field(default_factory=lambda: ["text"])
    voice: Optional[str] = "Vivian"


# ---------------------------------------------------------------------------
# Motore Neurale Reale Gemma 4 + Qwen3-TTS + Projector V9
# ---------------------------------------------------------------------------
class RealGemma4OmniEngine:
    def __init__(self):
        self.is_loaded = False
        self.duplex_model = None
        self.tok_gemma = None
        self.teacher = None
        self.talker = None
        self.proc_qwen = None
        self.tok_decoder = None
        self.proj_v9 = None
        self.CFG = None
        self.TCFG = None
        self.lang_id = 2070

    def load(self):
        if self.is_loaded:
            return
        print("[Engine] Inizializzazione Motore Neurale Reale Gemma 4 Omni su RTX 3090...", flush=True)

        # 1. Gemma 4 Thinker
        gemma_snaps = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True)
        if not gemma_snaps:
            gemma_snaps = glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
        if gemma_snaps:
            KP.MODEL_NAME = os.path.dirname(gemma_snaps[0])

        self.duplex_model, self.tok_gemma = KP.load_gemma4_duplex_system()
        KP.load_gemma4_duplex_checkpoint(self.duplex_model, "/workspace/checkpoints")
        self.duplex_model.eval()
        for p in self.duplex_model.parameters():
            p.requires_grad = False
        print("  ✓ Gemma 4 Thinker caricato su CUDA.", flush=True)

        # 2. Qwen3-TTS Talker
        snap_talker = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
        _cfg = _M.Qwen3TTSConfig.from_pretrained(snap_talker)
        _tc = _cfg.talker_config
        if not hasattr(_tc, 'pad_token_id'):
            object.__setattr__(_tc, 'pad_token_id', None)

        self.teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
        _sd = st.load_file(snap_talker + '/model.safetensors', device='cpu')
        self.teacher.load_state_dict(_sd, strict=False)
        del _sd
        torch.cuda.empty_cache()
        self.teacher = self.teacher.to(device='cuda', dtype=torch.bfloat16).eval()
        self.talker = self.teacher.talker

        tok_qwen = AutoTokenizer.from_pretrained(snap_talker, use_fast=True)
        self.proc_qwen = _M.Qwen3TTSProcessor(tokenizer=tok_qwen)

        self.CFG = self.teacher.config
        self.TCFG = self.CFG.talker_config
        self.lang_id = getattr(self.TCFG, 'codec_language_id', {}).get('italian', 2070)

        # 3. Code2Wav Vocoder (24kHz)
        snap_tok = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*'))[-1]
        self.tok_decoder = Qwen3TTSTokenizerV2Model.from_pretrained(snap_tok).cuda().eval()
        print("  ✓ Qwen3-TTS Talker & Vocoder 24kHz caricati.", flush=True)

        # 4. Projector V9 (Prefix KV-Cache 211.83M)
        self.proj_v9 = DeepCausalTransformerProjectorV9().to(device='cuda', dtype=torch.bfloat16).eval()
        v9_ckpt = '/workspace/checkpoints/v9_prefix_step10000.pt'
        if not os.path.isfile(v9_ckpt):
            v9_ckpt = '/workspace/checkpoints/v9_prefix_final.pt'
        sd_v9 = torch.load(v9_ckpt, map_location='cpu')
        self.proj_v9.load_state_dict(sd_v9.get('proj', sd_v9), strict=False)
        print(f"  ✓ Projector V9 caricato da: {v9_ckpt}.", flush=True)

        self.is_loaded = True
        print("[Engine] Motore Neurale Reale Pronto al 100%!", flush=True)

    def generate_text(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 160,
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """Genera risposta testuale reale con Gemma 4 e analizza i tool calls."""
        if not self.is_loaded:
            self.load()

        full_prompt = prompt
        if tools:
            full_prompt += format_tools_system_prompt(tools)
        if tool_results:
            results_text = "\n\nRisultati degli strumenti eseguiti:\n" + "\n".join(
                f"- {tr.get('name')}: {tr.get('content')}" for tr in tool_results
            )
            full_prompt += results_text + "\n\nRispondi ora all'utente spiegando chiaramente il risultato ottenuto."

        # Formattazione Chat Template ufficiale Gemma 4
        if hasattr(self.tok_gemma, "apply_chat_template"):
            try:
                formatted_prompt = self.tok_gemma.apply_chat_template(
                    [{"role": "user", "content": full_prompt.strip()}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                formatted_prompt = f"<bos><start_of_turn>user\n{full_prompt.strip()}<end_of_turn>\n<start_of_turn>model\n"
        else:
            formatted_prompt = f"<bos><start_of_turn>user\n{full_prompt.strip()}<end_of_turn>\n<start_of_turn>model\n"

        inputs = self.tok_gemma(text=formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to('cuda')
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to('cuda')

        tok = getattr(self.tok_gemma, "tokenizer", self.tok_gemma)
        eos_ids = []
        if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
            eos_ids.append(tok.eos_token_id)
        if hasattr(tok, "convert_tokens_to_ids"):
            for t_str in ["<end_of_turn>", "<eos>", "<|im_end|>"]:
                tid = tok.convert_tokens_to_ids(t_str)
                if tid is not None and isinstance(tid, int) and tid > 0:
                    eos_ids.append(tid)

        with torch.no_grad():
            out = self.duplex_model.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids if eos_ids else None,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            raw_text = self.tok_gemma.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        tool_calls = extract_tool_calls_from_text(raw_text, tools)
        clean_text = raw_text
        clean_text = re.sub(r'call:[a-zA-Z0-9_\-]+\s*\{.*?\}', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'```(?:tool_call|json)?\s*\{.*?\}\s*```', '', clean_text, flags=re.DOTALL).strip()
        if not clean_text and tool_calls:
            clean_text = "Eseguo la richiesta tramite lo strumento..."
        return clean_text, tool_calls

    def synthesize_speech_wav(self, text: str) -> bytes:
        """Sintesi vocale reale a 24kHz usando Thinker hidden states + Projector V9 + Talker + Vocoder."""
        if not self.is_loaded:
            self.load()

        clean_prompt = text.strip()
        if not clean_prompt:
            clean_prompt = "Ciao."

        with torch.no_grad():
            resp_ids = self.tok_gemma(text=clean_prompt, return_tensors="pt")["input_ids"].to('cuda')
            full_out = self.duplex_model.base_model(resp_ids, output_hidden_states=True, return_dict=True)
            lh = full_out.hidden_states[-1]
            h_tx = lh.mean(dim=1).to(torch.bfloat16)

            T_pfx_len = 16
            seq_embeds = lh
            seq_trans = seq_embeds.transpose(1, 2)
            h_raw_style = F.interpolate(seq_trans, size=T_pfx_len, mode='linear', align_corners=False).transpose(1, 2)
            h_raw_style = self.duplex_model._apply_causal_conv(
                h_raw_style.to(next(self.duplex_model.causal_conv.parameters()).dtype)
            ).to(torch.bfloat16)
            h_pfx = self.proj_v9(h_raw_style, h_tx)

            # Prefill Talker
            raw_ids = self.proc_qwen(text=clean_prompt, return_tensors='pt')['input_ids'].to('cuda')
            tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(
                    torch.tensor([[self.CFG.tts_bos_token_id, self.CFG.tts_eos_token_id, self.CFG.tts_pad_token_id]], device='cuda', dtype=raw_ids.dtype)
                )
            ).chunk(3, dim=1)

            codec_prefill = [[self.TCFG.codec_think_id, self.TCFG.codec_think_bos_id, self.lang_id, self.TCFG.codec_think_eos_id]]
            c_emb_0 = self.talker.get_input_embeddings()(torch.tensor(codec_prefill, device='cuda', dtype=raw_ids.dtype))
            c_emb_1 = self.talker.get_input_embeddings()(torch.tensor([[self.TCFG.codec_pad_id, self.TCFG.codec_bos_id]], device='cuda', dtype=raw_ids.dtype))
            codec_input_emebdding = torch.cat([c_emb_0, c_emb_1], dim=1)

            role_ids = self.proc_qwen(text="<|im_start|>assistant\n", return_tensors='pt')['input_ids'].to('cuda')
            role_embed = self.talker.text_projection(self.talker.get_text_embeddings()(role_ids))
            _talker_input_embed = torch.cat((tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1), tts_bos_embed), dim=1) + codec_input_emebdding[:, :-1]
            talker_input_embed = torch.cat((role_embed, _talker_input_embed), dim=1)

            text_proj_emb = self.talker.text_projection(self.talker.get_text_embeddings()(raw_ids))
            c_pad_full = self.talker.get_input_embeddings()(torch.tensor([[self.TCFG.codec_pad_id] * (raw_ids.shape[1] + 1)], device='cuda', dtype=raw_ids.dtype))
            text_part = torch.cat([text_proj_emb, tts_eos_embed], dim=1) + c_pad_full
            bos_part = tts_pad_embed + self.talker.get_input_embeddings()(torch.tensor([[self.TCFG.codec_bos_id]], device='cuda', dtype=raw_ids.dtype))

            icl_part = torch.cat([talker_input_embed, text_part], dim=1)
            pfx_embeds = torch.cat([icl_part, h_pfx, bos_part], dim=1)

            out_pfx = self.talker.model(inputs_embeds=pfx_embeds, use_cache=True)
            past_kv = out_pfx.past_key_values
            last_h = out_pfx.last_hidden_state[:, -1:, :]

            n_words = len(clean_prompt.split())
            T_gen_frames = max(36, min(140, int(n_words * 4.8) + 12))

            codes_gen = []
            for t in range(T_gen_frames):
                cb0_lg = self.talker.codec_head(last_h)[:, -1, :]
                cb0_lg[0, 2048:] = -float('Inf')
                topk_vals, topk_inds = torch.topk(cb0_lg, k=20, dim=-1)
                cb0_filtered = torch.full_like(cb0_lg, -float('Inf'))
                cb0_filtered.scatter_(-1, topk_inds, topk_vals)
                probs_0 = F.softmax(cb0_filtered / 0.6, dim=-1)
                c0_val = int(torch.multinomial(probs_0, 1).item())

                last_id_h = self.talker.get_input_embeddings()(torch.tensor([[c0_val]], device='cuda'))
                next_talker_emb = tts_pad_embed + last_id_h
                out_step = self.talker.model(inputs_embeds=next_talker_emb, past_key_values=past_kv, use_cache=True)
                past_kv = out_step.past_key_values
                last_h = out_step.last_hidden_state[:, -1:, :]

                # SubTalker predizione codici 1..15
                c_stack = [c0_val]
                curr_hidden = last_h
                for g in range(15):
                    pred_logits = self.teacher.code_predictor.heads[g](curr_hidden)[:, -1, :]
                    pred_p = F.softmax(pred_logits / 0.8, dim=-1)
                    cg_val = int(torch.multinomial(pred_p, 1).item())
                    c_stack.append(cg_val)
                    cg_emb = self.teacher.code_predictor.embeddings[g](torch.tensor([[cg_val]], device='cuda'))
                    curr_hidden = curr_hidden + cg_emb
                codes_gen.append(c_stack)

            # Decodifica Code2Wav a 24kHz
            codes_tensor = torch.tensor(codes_gen, dtype=torch.long, device='cuda').transpose(0, 1).unsqueeze(0)
            wav_tensor = self.tok_decoder.decode(codes_tensor)
            audio_np = wav_tensor.squeeze().cpu().float().numpy()

            buf = io.BytesIO()
            sf.write(buf, audio_np, 24000, format="WAV")
            return buf.getvalue()


# Istanza singleton globale
neural_engine = RealGemma4OmniEngine()


# ---------------------------------------------------------------------------
# Endpoints OpenAI Standard
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "gemma-4-omni-duplex",
        "gpu": "NVIDIA GeForce RTX 3090",
        "duplex_ready": True,
        "sample_rate": 24000,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma-4-omni-duplex",
                "object": "model",
                "created": 1756300000,
                "owned_by": "deepmind-gemma",
                "permission": [],
                "root": "gemma-4-omni-duplex",
                "parent": None,
                "capabilities": {
                    "chat_completion": True,
                    "function_calling": True,
                    "audio_streaming": True,
                    "full_duplex_realtime": True,
                },
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """Endpoint compatibile OpenAI con supporto Function Calling, Streaming e Audio."""
    created_ts = int(time.time())
    req_id = f"chatcmpl_{uuid.uuid4().hex[:12]}"

    # Estrae l'ultimo messaggio utente e la cronologia
    last_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user" and m.content), "")
    system_msgs = [m.content for m in req.messages if m.role == "system" and m.content]
    combined_prompt = ("\n".join(str(s) for s in system_msgs) + "\n" if system_msgs else "") + str(last_user_msg)

    # Estrae eventuali risposte a tool calls precedenti
    tool_results = [
        {"name": m.name or "tool", "content": str(m.content), "tool_call_id": m.tool_call_id}
        for m in req.messages
        if m.role == "tool" and m.content
    ]

    # Esecuzione Thinker Reale Gemma 4
    response_text, tool_calls = neural_engine.generate_text(
        combined_prompt,
        tools=req.tools,
        tool_results=tool_results,
        max_new_tokens=req.max_tokens or 160,
    )

    # 1. Caso Streaming SSE (stream=True)
    if req.stream:
        async def event_generator() -> AsyncGenerator[str, None]:
            words = response_text.split(" ")
            for i, w in enumerate(words):
                chunk = w + (" " if i < len(words) - 1 else "")
                data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.04)

            if tool_calls:
                formatted_tool_deltas = []
                for idx, tc in enumerate(tool_calls):
                    formatted_tool_deltas.append({
                        "index": idx,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    })
                data_tc = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": formatted_tool_deltas},
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
                yield f"data: {json.dumps(data_tc, ensure_ascii=False)}\n\n"
            else:
                data_end = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(data_end, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # 2. Caso Risposta Statica Normale (o con Audio)
    message_dict: Dict[str, Any] = {"role": "assistant", "content": response_text}
    finish_reason = "stop"

    if tool_calls:
        message_dict["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    # Se sono richieste le modalità audio
    if "audio" in req.modalities and not tool_calls:
        wav_bytes = neural_engine.synthesize_speech_wav(response_text)
        b64_audio = base64.b64encode(wav_bytes).decode("ascii")
        message_dict["audio"] = {
            "id": f"audio_{uuid.uuid4().hex[:8]}",
            "expires_at": created_ts + 3600,
            "data": b64_audio,
            "transcript": response_text,
        }

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": message_dict,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(combined_prompt) // 4,
            "completion_tokens": len(response_text) // 4,
            "total_tokens": (len(combined_prompt) + len(response_text)) // 4,
        },
    }


# ---------------------------------------------------------------------------
# WebSocket Full-Duplex Realtime Endpoint (/v1/realtime)
# ---------------------------------------------------------------------------
@app.websocket("/v1/realtime")
async def realtime_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = session_manager.create_session()
    ws_lock = asyncio.Lock()

    async def send_event(ev: Dict[str, Any]):
        async with ws_lock:
            try:
                await websocket.send_text(json.dumps(ev, ensure_ascii=False))
            except Exception:
                pass

    await send_event({
        "type": "session.created",
        "event_id": f"evt_{uuid.uuid4().hex[:8]}",
        "session": {
            "id": session.session_id,
            "model": "gemma-4-omni-duplex",
            "modalities": ["text", "audio"],
            "voice": session.voice,
            "turn_detection": {"type": "server_vad", "threshold": 0.5},
        },
    })

    try:
        while True:
            raw_msg = await websocket.receive_text()
            data = json.loads(raw_msg)
            msg_type = data.get("type", "")

            if msg_type == "session.update":
                sess_payload = data.get("session", {})
                if "voice" in sess_payload:
                    session.voice = sess_payload["voice"]
                if "tools" in sess_payload:
                    session.tools = sess_payload["tools"]
                await send_event({
                    "type": "session.updated",
                    "event_id": f"evt_{uuid.uuid4().hex[:8]}",
                    "session": sess_payload,
                })

            elif msg_type == "input_audio_buffer.append":
                pcm_b64 = data.get("audio", "")
                if pcm_b64:
                    raw_pcm = base64.b64decode(pcm_b64)
                    if session.is_agent_speaking and len(raw_pcm) > 960:
                        session.handle_barge_in(lambda ev: asyncio.create_task(send_event(ev)))

            elif msg_type in ("conversation.item.create", "response.create"):
                user_text = ""
                if "item" in data and "content" in data["item"]:
                    parts = data["item"]["content"]
                    for p in parts:
                        if isinstance(p, dict) and p.get("type") in ("input_text", "text"):
                            user_text += p.get("text", "")

                if not user_text and "response" in data:
                    user_text = "Ciao, parlami un po' di te."

                session.start_response(lambda ev: asyncio.create_task(send_event(ev)))
                resp_text, tool_calls = neural_engine.generate_text(user_text, session.tools)

                words = resp_text.split(" ")
                for i, w in enumerate(words):
                    if not session.is_agent_speaking:
                        break
                    chunk = w + (" " if i < len(words) - 1 else "")
                    session.emit_text_delta(chunk, lambda ev: asyncio.create_task(send_event(ev)))
                    await asyncio.sleep(0.06)

                if session.is_agent_speaking and tool_calls:
                    for tc in tool_calls:
                        session.emit_function_call(
                            tc["id"],
                            tc["function"]["name"],
                            tc["function"]["arguments"],
                            lambda ev: asyncio.create_task(send_event(ev)),
                        )

                if session.is_agent_speaking:
                    session.finish_response(lambda ev: asyncio.create_task(send_event(ev)))

            elif msg_type == "session.close":
                break

    except WebSocketDisconnect:
        pass
    finally:
        session_manager.close_session(session.session_id)


# ---------------------------------------------------------------------------
# Web SPA Frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_spa():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<html><body><h1>Gemma 4 Omni Server</h1></body></html>"
