# -*- coding: utf-8 -*-
"""
qwen_subtalker.py -- SUB-TALKER acustico Qwen3-TTS-12Hz in torch puro (ZERO transformers).

Replica 1:1 la struttura e la matematica del code predictor del teacher
Qwen/Qwen3-TTS-12Hz-1.7B-Base, come implementato nel pacchetto ufficiale
qwen-tts 0.1.1 (sorgente di riferimento: modeling_qwen3_tts.py):

  * Attenzione Qwen3TTSAttention con q_norm/k_norm RMSNorm PER-HEAD, RoPE dopo le
    norm, scaling head_dim^-0.5, GQA, bias=False        -> righe 885-1015
  * Qwen3TTSTalkerTextMLP (gate/up/down, SiLU, bias=False) -> righe 842-863
  * Qwen3TTSTalkerCodePredictorModel: 5 layer + norm finale + rotary_emb +
    codec_embedding = ModuleList(15 x Embedding(vocab 2048, dim=hidden TALKER))
                                                                  -> righe 1015-1120
  * small_to_mtp_projection (talker hidden -> cp hidden, bias=True) e
    lm_head = 15 x Linear(cp hidden, vocab 2048, bias=False)   -> righe 1156-1195
  * forward_finetune: logits[j] = lm_head[j](hidden[:, j+1]), j = 0..14
    (predice cb_{j+1} dalla posizione che ha IN INPUT cb_j)     -> righe 1197-1260
  * forward_sub_talker_finetune (input = [h_t] + codec_embedding(cb0) +
    code_predictor.get_input_embeddings()[i-1](cb_i) per i=1..14) -> righe 1612-1640

SCELTA VARIANTE (verificato): il teacher e' la variante **Base** (tts_model_type
"base"). A livello di modello COMPLETO il Base richiede un sample audio per la
clonazione (ref_audio Required in generate_voice_clone; speaker_encoder presente
nel checkpoint: 480 tensori). CustomVoice (9 preset, nessuno italiano nativo) e
VoiceDesign (prompt testuale) NON hanno speaker_encoder (405 tensori). QUI
extraiamo SOLO il sub-talker (code_predictor + codec_embedding di cb0), niente
speaker_encoder e niente main talker: l'input e' [h_t, codici] e la voce arriva da
h_t di Gemma (adapter trainabile) -- NESSUN sample vocale necessario a inferenza.
Nota: i pesi del sub-talker differiscono tra varianti (verificato via hash),
quindi la scelta Base e' vincolante per l'estrazione.

Config reali verificati (huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/raw/main/
config.json -> talker_config.code_predictor_config): hidden_size=1024,
num_hidden_layers=5, num_attention_heads=16, num_key_value_heads=8,
head_dim=128, rms_norm_eps=1e-06, hidden_act=silu, intermediate_size=3072,
attention_bias=False, vocab_size=2048, num_code_groups=16,
max_position_embeddings=65536, layer_types=[full_attention]*5.
Talker (1.7B): hidden_size=2048, vocab_size=3072 (codec_embedding del talker,
usato per cb0). RoPE: rope_theta default 10000.0, attenzione causale.

ALLINEAMENTO DELLA LOSS (decisione verificata, fonte tripla):
  * forward_sub_talker_finetune passa labels=codec_ids[:, 1:] con logits
    allineati per posizione (logits[j] <-> cb_{j+1}).
  * La loss_function di transformers 4.57.3 (ForCausalLMLoss in
    src/transformers/loss/loss_utils.py) APPLICA UNO SHIFT alle labels
    (pad + labels[..., 1:]) -> sbilancerebbe ogni codebook di una posizione.
  * Il generate del code predictor (lm_head[generation_steps] sulla posizione
    di input cb_{generation_steps-1}) e il report Qwen3-TTS sez. 3.2 (l'MTP
    "generates all residual codebooks" in catena RVQ) usano l'allineamento
    DIRETTO. L'unica interpretazione auto-coerente dei pesi rilasciati e' la
    catena RVQ: QUI usiamo l'allineamento diretto (nessuno shift), coerente
    train <-> inference.

Uso: il modulo e' CONGELATO (requires_grad_(False)) e usato dal training di
Gemma: input h_t (B,T,2048) dall'adapter + codec_ids GT (B,T,16); la loss di
coupling fa fluire i gradienti a Gemma attraverso i pesi congelati
(attivazioni, non pesi). Guard fail-fast: cb0 dei frame validi deve essere
< cb0_vocab_size (3072); valori >= 2048 generano un warning (anomalia dati).

NOTA MEMORIA: ~109 tensori, ~400MB bf16 per la variante 1.7B (pesi congelati).
"""

import json
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Costituenti (replica esatta di modeling_qwen3_tts.py)
# ---------------------------------------------------------------------------
class Qwen3TTSRMSNorm(nn.Module):
    """RMSNorm T5-style: calcolo in fp32, eps config (default 1e-6)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3TTSRotaryEmbedding(nn.Module):
    """RoPE standard (rope_type='default'): emb = cat(freqs, freqs), theta 10000."""

    def __init__(self, head_dim: int, max_position_embeddings: int = 65536, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # identico al sorgente: calcolo in fp32 forzato (autocast off)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3TTSTalkerTextMLP(nn.Module):
    """gate/up/down projection con SiLU (bias=False), come il sorgente."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3TTSDecoderLayer(nn.Module):
    """Layer decoder del code predictor: pre-norm RMSNorm + attenzione con
    q_norm/k_norm RMSNorm PER-HEAD (head_dim) + RoPE + GQA + MLP SiLU."""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.head_dim = cfg["head_dim"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.eps = cfg["rms_norm_eps"]

        self.input_layernorm = Qwen3TTSRMSNorm(self.hidden_size, eps=self.eps)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.self_attn.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.self_attn.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.self_attn.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.self_attn.q_norm = Qwen3TTSRMSNorm(self.head_dim, eps=self.eps)
        self.self_attn.k_norm = Qwen3TTSRMSNorm(self.head_dim, eps=self.eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(self.hidden_size, eps=self.eps)
        self.mlp = Qwen3TTSTalkerTextMLP(self.hidden_size, cfg["intermediate_size"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)

        # q/k/v: view (B, S, n_heads, head_dim) -> transpose(1,2) -> (B, n_heads, S, head_dim)
        input_shape = h.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.self_attn.q_norm(self.self_attn.q_proj(h).view(hidden_shape)).transpose(1, 2)
        key_states = self.self_attn.k_norm(self.self_attn.k_proj(h).view(hidden_shape)).transpose(1, 2)
        value_states = self.self_attn.v_proj(h).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # GQA: espandi i kv-head al numero di query-head (repeat_kv del sorgente)
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(h.dtype)
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        attn_output = self.self_attn.o_proj(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


# ---------------------------------------------------------------------------
# Modulo principale (replica di Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
# nel percorso forward_sub_talker_finetune)
# ---------------------------------------------------------------------------
class Qwen3TTSSubTalker(nn.Module):
    """Sub-talker acustico Qwen3-TTS-12Hz congelato (code predictor + embedding cb0).

    forward_sub_talker_finetune(h_t, codec_ids):
        h_t        (B, T, talker_hidden)  -> attivazione per-frame di Gemma (post adapter)
        codec_ids  (B, T, 16)             -> codici GT (cb0..cb15); PAD_TARGET_ID (-100)
                                              sui frame di padding
        ritorna logits (B, T, 15, cp_vocab): logits[:, :, j] = predizione di cb_{j+1}
                                              (allineamento RVQ diretto, no shift)

    coupling_loss(logits, codec_ids, frame_weights):
        CE diretta su logits[:, :, j] <-> codec_ids[:, :, j+1] (j=0..14), nessuno
        shift, ignore dei frame pad, scalata per frame_weights.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = dict(cfg)
        cp = self.cfg
        # Embedding di cb0: codec_embedding del TALKER (3072 x talker_hidden),
        # verificato: forward_sub_talker_finetune usa self.get_input_embeddings()
        # del talker (spazio 3072, NON 2048).
        self.cb0_embedding = nn.Embedding(cp["cb0_vocab_size"], cp["talker_hidden_size"])
        # Embedding di gruppo: code_predictor.model.codec_embedding
        # (15 x Embedding(vocab 2048, dim = talker_hidden)), vedi riga 1019 del sorgente:
        # Qwen3TTSTalkerCodePredictorModel(config, talker_config.hidden_size)
        self.group_embeddings = nn.ModuleList(
            [nn.Embedding(cp["vocab_size"], cp["talker_hidden_size"]) for _ in range(cp["num_code_groups"] - 1)]
        )
        # Proiezione talker_hidden -> cp_hidden (bias=True); Identity se uguali
        if cp["talker_hidden_size"] != cp["hidden_size"]:
            self.small_to_mtp_projection = nn.Linear(cp["talker_hidden_size"], cp["hidden_size"], bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()
        self.layers = nn.ModuleList([Qwen3TTSDecoderLayer(cp) for _ in range(cp["num_hidden_layers"])])
        self.norm = Qwen3TTSRMSNorm(cp["hidden_size"], eps=cp["rms_norm_eps"])
        self.rotary_emb = Qwen3TTSRotaryEmbedding(
            cp["head_dim"], max_position_embeddings=cp["max_position_embeddings"], theta=cp["rope_theta"]
        )
        # 15 lm_head (bias=False): lm_head[j] predice cb_{j+1} alla posizione j+1
        self.lm_head = nn.ModuleList(
            [nn.Linear(cp["hidden_size"], cp["vocab_size"], bias=False) for _ in range(cp["num_code_groups"] - 1)]
        )
        self.register_buffer(
            "_causal_mask", self._build_causal_mask(cp["num_code_groups"]), persistent=False
        )
        self._num_warned_ge2048 = 0

    @staticmethod
    def _build_causal_mask(seq_len: int) -> torch.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(torch.float32).min)
        mask[0, 0, :, :].triu_(1)  # -inf sopra la diagonale -> causale
        return mask

    def _forward_flat(self, h_t_flat: torch.Tensor, codec_ids_flat: torch.Tensor) -> torch.Tensor:
        """Code predictor su un batch piatto (B*T, 16, ...): replica esatta della
        forward del teacher con inputs_embeds concatenati (h_t, cb0, cb1..cb14)."""
        B, _, _ = h_t_flat.shape  # batch piatto (B*T, 1, talker_hidden): S reale = lunghezza dopo la concatenazione
        inputs = [h_t_flat]  # posizione 0: h_t (senza embedding)
        # posizione 1: cb0 tramite codec_embedding del talker (spazio cb0_vocab)
        inputs.append(self.cb0_embedding(codec_ids_flat[:, :1].clamp(min=0)))
        # posizioni 2..15: cb1..cb14 tramite gli embedding di gruppo 0..13
        for g in range(self.cfg["num_code_groups"] - 2):
            inputs.append(self.group_embeddings[g](codec_ids_flat[:, g + 1 : g + 2].clamp(min=0)))
        inputs_embeds = torch.cat(inputs, dim=1)  # (B, 16, talker_hidden)
        S = inputs_embeds.shape[1]  # 16 posizioni (h_t, cb0, cb1..cb14): FIX RoPE (prima S=1 da h_t_flat)
        hidden_states = self.small_to_mtp_projection(inputs_embeds)  # (B, 16, cp_hidden)

        position_ids = torch.arange(0, S, device=h_t_flat.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = self._causal_mask.to(h_t_flat.device).to(h_t_flat.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, causal_mask)
        hidden_states = self.norm(hidden_states)

        # logits[j] = lm_head[j](hidden[:, j+1]) per j=0..14 (forward_finetune,
        # righe 1235-1245 del sorgente) -> predice cb_{j+1} (RVQ chain, no shift)
        logits = torch.stack(
            [self.lm_head[j](hidden_states[:, j + 1]) for j in range(self.cfg["num_code_groups"] - 1)], dim=1
        )  # (B, 15, cp_vocab)
        return logits

    def forward_sub_talker_finetune(self, h_t: torch.Tensor, codec_ids: torch.Tensor) -> torch.Tensor:
        """Percorso teacher-forcing del coupling (replica di forward_sub_talker_finetune).

        h_t (B, T, talker_hidden), codec_ids (B, T, 16) -> (B, T, 15, cp_vocab).
        Guard fail-fast: cb0 dei frame VALIDI deve essere < cb0_vocab_size.
        """
        B, T, _ = h_t.shape
        if h_t.shape[2] != self.cfg["talker_hidden_size"]:
            raise ValueError(
                f"h_t con hidden {h_t.shape[2]}, atteso {self.cfg['talker_hidden_size']} "
                f"(usa l'adapter h_t -> talker_hidden)."
            )
        if codec_ids.shape[2] != self.cfg["num_code_groups"]:
            raise ValueError(
                f"codec_ids con {codec_ids.shape[2]} gruppi, attesi {self.cfg['num_code_groups']}."
            )

        cb0 = codec_ids[:, :, 0]
        valid = cb0 >= 0
        if valid.any():
            mx = int(cb0[valid].max().item())
            if mx >= self.cfg["cb0_vocab_size"]:
                raise AssertionError(
                    f"cb0={mx} >= cb0_vocab_size={self.cfg['cb0_vocab_size']}: il codec_embedding "
                    "del talker non copre questo valore. Dati di Fase 1 anomali (range atteso [0, 2048))."
                )
            if mx >= self.cfg["vocab_size"]:
                self._num_warned_ge2048 += 1
                if self._num_warned_ge2048 <= 3:
                    print(
                        f"[QwenSubTalker] [WARN] cb0 max = {mx} (>= vocab {self.cfg['vocab_size']}): "
                        "valore fuori dal range semantico del tokenizer 12Hz (codebook_size 2048). "
                        "Embedding 3072 lo copre, ma e' un'anomalia dati da investigare."
                    )

        h_flat = h_t.reshape(B * T, 1, self.cfg["talker_hidden_size"])
        c_flat = codec_ids.reshape(B * T, self.cfg["num_code_groups"])
        logits = self._forward_flat(h_flat, c_flat)  # (B*T, 15, cp_vocab)
        return logits.reshape(B, T, self.cfg["num_code_groups"] - 1, self.cfg["vocab_size"])

    def coupling_loss(
        self,
        logits: torch.Tensor,
        codec_ids: torch.Tensor,
        frame_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Loss NLL accoppiata: CE diretta (nessuno shift) su logits[:, :, j] <-> cb_{j+1}.

        I frame di padding (PAD_TARGET_ID=-100) sono ignorati (ignore_index + mask
        esplicita sul denominatore), coerente con compute_discoupled_loss della
        pipeline. Se presente, frame_weights scala per-frame (come per cb0/acustici).
        """
        device = logits.device
        V = self.cfg["vocab_size"]
        labels = codec_ids[:, :, 1:]  # (B, T, 15)
        if labels.device != device:
            labels = labels.to(device)
        per = F.cross_entropy(
            logits.reshape(-1, V), labels.reshape(-1), reduction="none", ignore_index=PAD_TARGET_ID
        ).reshape(*labels.shape)
        if frame_weights is not None:
            fw = frame_weights.to(device)
            per = per * fw.unsqueeze(-1)
        valid = (labels != PAD_TARGET_ID).float()
        denom = valid.sum().clamp(min=1)
        if denom <= 0:
            return torch.zeros((), device=device, dtype=logits.dtype)
        return (per * valid).sum() / denom

    def sample_acoustic_codes(
        self,
        h_t: torch.Tensor,
        cb0: torch.Tensor,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        warn_cb0_clamp: bool = True,
    ) -> torch.Tensor:
        """Inferenza: dato h_t (B, T, talker_hidden) e cb0 (B, T) dal semantic_head,
        genera i 15 codebook residui cb1..cb15 tramite il sub-talker causale congelato.

        Protocollo ufficiale Qwen3-TTS (SDD):
          - modeling_qwen3_tts.py (righe 1652-1677): code_predictor.generate(...) con
            do_sample, top_p, top_k, temperature.
          - qwen3_tts_model.py (righe 320-352): hard_defaults ufficiali del sub-talker:
            subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0,
            subtalker_temperature=0.9.

        Args:
            h_t: (B, T, talker_hidden_size) stati nascosti (da adapter o teacher nativo).
            cb0: (B, T) codici semantici del primo codebook (range [0, vocab_size)).
            temperature: temperatura di scaling dei logit. Se <= 0.0, esegue argmax puro
                (greedy). Se > 0.0, esegue sampling stocastico con top_k e top_p.
            top_k: numero massimo di candidati ad alta probabilita' da considerare (default 50).
                Se top_k <= 0 o top_k >= vocab_size, il filtro non viene applicato.
                Se top_k == 1, coincide deterministicamente con argmax.
            top_p: soglia di probabilita' cumulativa per il nucleus sampling (default 1.0 = disattivo).
            seed: seed casuale opzionale per riproducibilita' deterministica del sampling.
            device: torch.device di destinazione (se None, usa cb0.device).
            warn_cb0_clamp: se True, stampa un avviso qualora cb0 superi vocab_size - 1.

        Returns:
            torch.Tensor: (B, T, 15) codici acustici campionati (cb1..cb15).
        """
        if seed is not None:
            torch.manual_seed(seed)

        B, T = cb0.shape
        dev = device if device is not None else cb0.device
        cb0 = cb0.to(dev).long()
        # cb0 dal semantic_head di Gemma ha vocabolario 4096: clamp nel range
        # semantico GT [0, 2048) per non uscire dalla distribuzione del teacher.
        if cb0.numel() > 0 and cb0.max() >= self.cfg["vocab_size"]:
            if warn_cb0_clamp:
                print(
                    f"[QwenSubTalker] [WARN] clamp cb0 campionato: max={int(cb0.max().item())} -> "
                    f"{self.cfg['vocab_size'] - 1} (semantic_head 4096 vs range GT 2048)."
                )
            cb0 = cb0.clamp(max=self.cfg["vocab_size"] - 1)

        h_flat = h_t.reshape(B * T, 1, self.cfg["talker_hidden_size"])
        full = torch.zeros(B, T, self.cfg["num_code_groups"], dtype=torch.long, device=dev)
        full[:, :, 0] = cb0
        seq = cb0.unsqueeze(-1)  # (B, T, 1)
        V = self.cfg["vocab_size"]
        for g in range(1, self.cfg["num_code_groups"]):
            # full: col0=cb0, col1..g-1 = codici gia' campionati, resto 0 (junk:
            # la causalita' rende innocue le posizioni future per la posizione g)
            if g > 1:
                full[:, :, 1:g] = seq[:, :, 1:]
            logits = self._forward_flat(h_flat, full.reshape(B * T, self.cfg["num_code_groups"]))  # (B*T, 15, V)
            lg = logits[:, g - 1, :].reshape(B, T, V)  # predice cb_g

            if temperature <= 0.0:
                sampled = lg.argmax(dim=-1)
            else:
                logits_step = lg / float(temperature)

                # 1. Filtro Top-K (HF style)
                if top_k is not None and 0 < top_k < V:
                    topk_vals, _ = torch.topk(logits_step, min(int(top_k), V), dim=-1)
                    kth_val = topk_vals[..., -1, None]
                    logits_step = torch.where(
                        logits_step < kth_val,
                        torch.tensor(float("-inf"), device=logits_step.device, dtype=logits_step.dtype),
                        logits_step,
                    )

                # 2. Filtro Top-P / Nucleus (HF style)
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_step, descending=True, dim=-1)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Maschera token sopra la soglia cumulativa top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift a destra di 1 per mantenere almeno il primo token che sfora
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    logits_step = logits_step.masked_fill(indices_to_remove, float("-inf"))

                probs = F.softmax(logits_step, dim=-1)
                flat_probs = probs.reshape(-1, V)
                sampled = torch.multinomial(flat_probs, num_samples=1).reshape(B, T)

            seq = torch.cat([seq, sampled.unsqueeze(-1)], dim=-1)
        return seq[:, :, 1:]  # (B, T, 15)


# ---------------------------------------------------------------------------
# Caricamento dal file slim (estratto da extract_qwen_subtalker.py)
# ---------------------------------------------------------------------------
PAD_TARGET_ID = -100  # coerente con kaggle_pipeline.py

DEFAULT_CONFIG = {
    "talker_hidden_size": 2048,
    "hidden_size": 1024,
    "num_hidden_layers": 5,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "vocab_size": 2048,
    "num_code_groups": 16,
    "max_position_embeddings": 65536,
    "rope_theta": 10000.0,
    "cb0_vocab_size": 3072,
}


def load_qwen_subtalker(subtalker_dir: str, device: Optional[torch.device] = None, dtype=torch.bfloat16):
    """Carica il modulo dal dir (qwen_subtalker.safetensors + config.json).

    Ritorna (module, cfg_dict). Il modulo viene restituito congelato (eval +
    requires_grad_(False)): solo l'adapter h_t di Gemma e' addestrabile.
    """
    import safetensors.torch as st

    cfg_path = os.path.join(subtalker_dir, "config.json")
    st_path = os.path.join(subtalker_dir, "qwen_subtalker.safetensors")
    if not os.path.exists(cfg_path) or not os.path.exists(st_path):
        raise FileNotFoundError(
            f"Sub-talker non trovato in '{subtalker_dir}' (attesi config.json e "
            "qwen_subtalker.safetensors). Esegui prima extract_qwen_subtalker.py."
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = {**DEFAULT_CONFIG, **cfg}
    model = Qwen3TTSSubTalker(cfg)
    state = st.load_file(st_path)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    if device is not None:
        model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)
    return model, cfg


# ---------------------------------------------------------------------------
# Self-test comportamentali (TDD): shape, causalita', gradienti, mask, guard
# ---------------------------------------------------------------------------
def _selftest() -> int:
    torch.manual_seed(0)
    cfg = dict(DEFAULT_CONFIG)
    model = Qwen3TTSSubTalker(cfg).eval()
    failures = []

    def check(name: str, cond: bool, extra: str = ""):
        status = "OK" if cond else "FAIL"
        print(f"  [{status}] {name}{(' | ' + extra) if extra else ''}")
        if not cond:
            failures.append(name)

    B, T = 2, 8
    h_t = torch.randn(B, T, cfg["talker_hidden_size"])
    codes = torch.randint(0, cfg["vocab_size"], (B, T, cfg["num_code_groups"]))
    codes[:, 3, :] = PAD_TARGET_ID  # frame di padding

    # 1) Forma dell'output
    logits = model.forward_sub_talker_finetune(h_t, codes)
    check(
        "shape logits (B,T,15,V)",
        tuple(logits.shape) == (B, T, cfg["num_code_groups"] - 1, cfg["vocab_size"]),
        str(tuple(logits.shape)),
    )

    # 2) Loss finita + mask dei frame pad (frame 3 deve essere escluso)
    loss = model.coupling_loss(logits, codes)
    check("loss finita e non NaN", torch.isfinite(loss).item() and loss.numel() == 1)
    codes_all_pad = torch.full_like(codes, PAD_TARGET_ID)
    loss_pad = model.coupling_loss(logits, codes_all_pad)
    check("loss su frame tutti pad = 0", loss_pad.item() == 0.0, f"{loss_pad.item():.6f}")

    # 3) Determinismo
    logits2 = model.forward_sub_talker_finetune(h_t, codes)
    check("determinismo", torch.equal(logits, logits2))

    # 4) Proprieta' causale: logits[:, :, j] non dipende da codici col k > j
    #    (posizione k+1 oltre la finestra causale di posizione j+1; cb15 non e' mai input)
    codes_pert = codes.clone()
    codes_pert[:, :, 15] = (codes_pert[:, :, 15] + 1) % cfg["vocab_size"]
    logits_pert = model.forward_sub_talker_finetune(h_t, codes_pert)
    check("logits j=0 invariante a cb15 (causalita')", torch.equal(logits[:, :, 0], logits_pert[:, :, 0]))
    check("logits j=14 invariante a cb15", torch.equal(logits[:, :, 14], logits_pert[:, :, 14]))

    # 5) Gradienti: la loss di coupling fluisce fino a h_t (input) e all'adapter
    h_t2 = torch.randn(B, T, cfg["talker_hidden_size"], requires_grad=True)
    lg = model.forward_sub_talker_finetune(h_t2, codes)
    lo = model.coupling_loss(lg, codes)
    lo.backward()
    check("gradiente fluisce a h_t", h_t2.grad is not None and bool(torch.isfinite(h_t2.grad).all()))

    # 6) Guard fail-fast su cb0 fuori range (3072)
    codes_bad = codes.clone()
    codes_bad[:, 0, 0] = cfg["cb0_vocab_size"] + 10
    raised = False
    try:
        model.forward_sub_talker_finetune(h_t, codes_bad)
    except AssertionError:
        raised = True
    check("guard cb0 >= cb0_vocab_size alza AssertionError", raised)

    # 7) Sampling inferenza: (B, T, 15), range [0, 2048), determinista con temp=0
    cb0 = torch.randint(0, cfg["vocab_size"], (B, T))
    ac = model.sample_acoustic_codes(h_t, cb0, temperature=0.0)
    check("sample_acoustic_codes shape (B,T,15)", tuple(ac.shape) == (B, T, 15), str(tuple(ac.shape)))
    check("sample in range [0, 2048)", bool((ac >= 0).all() and (ac < cfg["vocab_size"]).all()))
    # clamp del cb0 campionato fuori range (semantic_head 4096)
    cb0_big = torch.full((B, T), 3500, dtype=torch.long)
    ac2 = model.sample_acoustic_codes(h_t, cb0_big, temperature=0.0, warn_cb0_clamp=False)
    check("clamp cb0 campionato 3500 -> range", bool(ac2.shape[2] == 15))

    print(f"\nSelftest: {len(failures)} fallimenti su 9 controlli.")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    else:
        print("Uso: python qwen_subtalker.py --selftest")
